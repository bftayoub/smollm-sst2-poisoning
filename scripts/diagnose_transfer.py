"""Read existing checkpoints and measure poison/clean gradient geometry."""
import argparse
import gc
import random
from pathlib import Path

import torch

from poisonlab.common import Cost, file_digest, read_json, seed_all, write_json
from poisonlab.model import load_model
from poisonlab.poison import cosine, discrete_gradient, dot, gradient, parameters_for_alignment
from poisonlab.train import predict


def geometry(target, other):
    target_norm, other_norm = float(dot(target, target).sqrt()), float(dot(other, other).sqrt())
    return {"cosine": float(cosine(target, other)), "target_norm": target_norm,
            "gradient_norm": other_norm, "dot_with_target": float(dot(target, other)),
            "projection_on_target_unit": float(dot(target, other))/max(target_norm, 1e-12)}


def margin_gradient(model, codec, params, target):
    device = next(model.parameters()).device
    batch = codec.batch([target], device, labeled=False)
    logits = model(**batch).logits[0, -1]
    margin = logits[codec.label_ids[target["label"]]]-logits[codec.label_ids[target["desired_label"]]]
    grads = torch.autograd.grad(margin, params, allow_unused=True)
    return float(margin.detach()), tuple(torch.zeros_like(p) if g is None else g.detach()
                                        for p, g in zip(params, grads))


def diagnose(root, output, device="cuda", clean_samples=256, checkpoints=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError("Write diagnostics outside the source run to preserve original evidence")
    cfg = read_json(root / "config.json")
    cfg["device"] = device
    seed_all(cfg["reference_seed"], cfg["threads"])
    parts = read_json(root / "partitions.json")
    if clean_samples < 0:
        raise ValueError("clean_samples must be nonnegative; zero means all training rows")
    clean = parts["train"]
    if clean_samples and clean_samples < len(clean):
        clean = random.Random(cfg["split_seed"]).sample(clean, clean_samples)
    targets = read_json(root / "targets.json")
    report = {"source_run": str(root), "device": device,
              "source_config_sha256": file_digest(root / "config.json"),
              "clean_gradient_is_sample_estimate": len(clean) != len(parts["train"]),
              "clean_sample_ids": [r["id"] for r in clean],
              "clean_sample_count": len(clean), "clean_total_count": len(parts["train"]),
              "interpretation": "First-order SGD geometry; not a prediction of the cumulative AdamW update",
              "checkpoints": []}
    checkpoints = checkpoints or ["initial", "reference"]
    for checkpoint in checkpoints:
        path = (root / checkpoint).resolve()
        if root not in path.parents or not (path / "model.safetensors").is_file():
            raise ValueError(f"Checkpoint must be a saved model inside source run: {checkpoint}")
        model, codec = load_model(cfg, root, checkpoint)
        model.eval()
        names, params = parameters_for_alignment(model, cfg["poison"]["alignment_parameters"])
        cost = Cost(next(model.parameters()).device)
        clean_gradient = discrete_gradient(model, codec, params, clean, cfg["train"]["batch_size"])
        record = {"checkpoint": checkpoint,
                  "weights_sha256": file_digest(path / "model.safetensors"),
                  "alignment_parameter_count": sum(p.numel() for p in params), "targets": []}
        for target in targets:
            wanted = {**target, "label": target["desired_label"]}
            gt = tuple(g.detach() for g in gradient(model, params, codec.batch([wanted], next(model.parameters()).device)))
            margin, gm = margin_gradient(model, codec, params, target)
            campaign = root / "campaigns" / target["id"].replace(":", "_")
            entry = {"target_id": target["id"], "target_prediction": predict(model, codec, [target])[0],
                     "desired_label": target["desired_label"],
                     "true_minus_desired_logit_margin": margin,
                     "clean_training_gradient": geometry(gt, clean_gradient),
                     "clean_gradient_vs_margin": geometry(gm, clean_gradient), "conditions": {}}
            for condition in ("clean", "random", "optimized"):
                rows_path = campaign / f"seed_{cfg['victim_seeds'][0]}" / condition / "added_training_rows.json"
                rows = read_json(rows_path)
                gp = discrete_gradient(model, codec, params, rows, cfg["poison"]["microbatch"])
                fraction = len(rows)/(len(parts["train"])+len(rows))
                mix = tuple((1-fraction)*a+fraction*b for a, b in zip(clean_gradient, gp))
                entry["conditions"][condition] = {
                    "added_rows_sha256": file_digest(rows_path), "added_fraction": fraction,
                    "added_gradient": geometry(gt, gp), "mixed_training_gradient": geometry(gt, mix),
                    "added_gradient_vs_margin": geometry(gm, gp),
                    "mixed_gradient_vs_margin": geometry(gm, mix)}
                del gp, mix
            record["targets"].append(entry)
            del gt, gm
        record["cost"] = cost.finish()
        report["checkpoints"].append(record)
        write_json(output / "transfer_diagnostics.json", report)
        print(f"Measured {checkpoint}: {record['targets']}", flush=True)
        del clean_gradient, params, names, model, codec
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    report["status"] = "complete"
    write_json(output / "transfer_diagnostics.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clean-samples", type=int, default=256)
    parser.add_argument("--checkpoints", nargs="+", default=["initial", "reference"])
    args = parser.parse_args()
    diagnose(args.run_dir, args.output_dir, args.device, args.clean_samples, args.checkpoints)
