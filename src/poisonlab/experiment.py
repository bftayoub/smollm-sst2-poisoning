import csv
import gc
import traceback
from pathlib import Path

import torch

from .common import digest, environment, file_digest, read_json, seed_all, write_json
from .data import assert_no_targets, load_parts, prepare
from .model import load_model, prepare_initial
from .poison import optimize
from .train import choose_targets, clean_additions, metrics, predict, train_model


def bind_run(cfg, root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.json"
    if path.exists() and digest(read_json(path)) != digest(cfg):
        raise ValueError("Run directory already has a different config. Use a new directory.")
    write_json(path, cfg)
    if not (root / "environment.json").exists():
        write_json(root / "environment.json", environment())
    current_environment = environment()
    write_json(root / "environments" / (digest(current_environment) + ".json"), current_environment)


def release(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def execute_stage(root, name, action):
    status_path = root / "stages" / (name + ".json")
    if status_path.exists() and read_json(status_path)["status"] == "complete":
        print(f"skip completed stage {name}", flush=True)
        return
    write_json(status_path, {"status": "running"})
    try:
        action()
    except Exception as exc:
        write_json(status_path, {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        raise
    write_json(status_path, {"status": "complete"})


def prepare_stage(cfg, root):
    prepare(cfg, root)
    seed_all(cfg["reference_seed"], cfg["threads"])
    prepare_initial(cfg, root)


def load_targets(root):
    targets = read_json(root / "targets.json")
    manifest = root / "targets_manifest.json"
    if manifest.exists() and digest(targets) != read_json(manifest)["targets_sha256"]:
        raise ValueError("Fixed target selection changed; use a new experiment directory")
    return targets


def reference_stage(cfg, root):
    parts = load_parts(root)
    seed_all(cfg["reference_seed"], cfg["threads"])
    model, codec = load_model(cfg, root)
    train_model(model, codec, parts["train"], cfg, cfg["reference_seed"], root / "reference", parts["reserve"])
    preds = predict(model, codec, parts["dev"], cfg["evaluation_batch_size"])
    write_json(root / "reference" / "dev_predictions.json", preds)
    write_json(root / "reference" / "dev_metrics.json", metrics(preds))
    choose_targets(model, codec, parts["reserve"], cfg, root)


def poison_stage(cfg, root, target):
    parts = load_parts(root)
    targets = load_targets(root)
    assert_no_targets(parts["train"], targets)
    model, codec = load_model(cfg, root, "reference")
    campaign = root / "campaigns" / target["id"].replace(":", "_")
    optimize(model, codec, target, cfg, campaign / "poisons", targets)


def victim_stage(cfg, root, target, seed, condition):
    parts = load_parts(root)
    targets = load_targets(root)
    campaign = root / "campaigns" / target["id"].replace(":", "_")
    if condition == "clean":
        additions = clean_additions(parts["train"], cfg["poison"]["count"], cfg["poison_seed"])
    else:
        status = root / "stages" / ("poison_" + target["id"].replace(":", "_") + ".json")
        if not status.exists() or read_json(status)["status"] != "complete":
            raise RuntimeError("Synthetic victims require a completed poison campaign")
        additions = read_json(campaign / "poisons" / f"{condition}.json")
    assert len(additions) == cfg["poison"]["count"]
    assert sum(r["label"] == 1 for r in additions) == len(additions)//2
    assert_no_targets(parts["train"]+additions, targets)
    output = campaign / f"seed_{seed}" / condition
    seed_all(seed, cfg["threads"])
    # Every victim starts here, NEVER at the reference checkpoint.
    model, codec = load_model(cfg, root, "initial")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "added_training_rows.json", additions)
    cost = train_model(model, codec, parts["train"]+additions, cfg, seed, output, targets)
    general = predict(model, codec, parts["general"], cfg["evaluation_batch_size"])
    target_pred = predict(model, codec, [target], cfg["evaluation_batch_size"])[0]
    result = {"status": "complete", "target_id": target["id"], "seed": seed, "condition": condition,
              "target": target_pred, "desired_label": target["desired_label"],
              "attack_success": target_pred["prediction"] == target["desired_label"],
              "general": metrics(general), "cost": cost, "initial_checkpoint": "initial",
              "model_revision": read_json(root / "model_manifest.json")["revision"],
              "initial_weights_sha256": file_digest(root / "initial" / "model.safetensors")}
    write_json(output / "general_predictions.json", general)
    write_json(output / "result.json", result)


def evaluate(cfg, root):
    targets = load_targets(root)
    results = []
    for target in targets:
        campaign = root / "campaigns" / target["id"].replace(":", "_")
        for seed in cfg["victim_seeds"]:
            paired = {}
            for condition in ("clean", "random", "optimized"):
                path = campaign / f"seed_{seed}" / condition / "result.json"
                row = read_json(path) if path.exists() else {
                    "status": "missing", "target_id": target["id"], "seed": seed, "condition": condition}
                status = root / "stages" / f"victim_{target['id'].replace(':', '_')}_{seed}_{condition}.json"
                if status.exists() and read_json(status)["status"] != "complete":
                    row = {**row, **read_json(status)}
                paired[condition] = row
            clean = paired["clean"]
            for condition, row in paired.items():
                if row["status"] == "complete" and clean["status"] == "complete":
                    wanted = row["desired_label"]
                    row["clean_target_prediction"] = clean["target"]["prediction"]
                    row["clean_desired_probability"] = clean["target"]["probabilities"][wanted]
                    row["desired_probability_delta"] = row["target"]["probabilities"][wanted] - row["clean_desired_probability"]
                    row["target_flip_from_correct_clean"] = (
                        clean["target"]["prediction"] == target["label"] and row["attack_success"])
                    row["attack_success_delta_vs_clean"] = int(row["attack_success"])-int(clean["attack_success"])
                results.append(row)
    # Validate comparability from recorded evidence, not assumed settings.
    for target in targets:
        for seed in cfg["victim_seeds"]:
            group = [r for r in results if r["target_id"] == target["id"] and r["seed"] == seed and r["status"] == "complete"]
            if len(group) == 3:
                for key in ("row_count", "updates", "added_example_exposures"):
                    if len({r["cost"][key] for r in group}) != 1:
                        raise ValueError(f"Unpaired condition budget: {key}")
                if len({r["initial_weights_sha256"] for r in group}) != 1:
                    raise ValueError("Victim initial checkpoint weights differ")
                campaign = root / "campaigns" / target["id"].replace(":", "_")
                orders = [read_json(campaign / f"seed_{seed}" / r["condition"] / "training_order.json") for r in group]
                if orders[1:] != [orders[0], orders[0]]:
                    raise ValueError("Victim training order differs across conditions")
                random_rows = read_json(campaign / "poisons" / "random.json")
                opt_rows = read_json(campaign / "poisons" / "optimized.json")
                for a, b in zip(random_rows, opt_rows):
                    for field in ("retokenized_review_ids", "retokenized_prompt_ids"):
                        if len(a[field]) != len(b[field]):
                            raise ValueError("Synthetic lengths not paired")
    summary = {}
    for condition in ("clean", "random", "optimized"):
        group = [r for r in results if r["condition"] == condition and r["status"] == "complete"]
        expected = len(targets)*len(cfg["victim_seeds"])
        summary[condition] = {"completed": len(group), "expected": expected,
                              "missing_or_failed": expected-len(group),
                              "attack_success_rate_completed_only": sum(r["attack_success"] for r in group)/len(group) if group else None,
                              "mean_general_accuracy": sum(r["general"]["accuracy"] for r in group)/len(group) if group else None,
                              "mean_general_macro_f1": sum(r["general"]["macro_f1"] for r in group)/len(group) if group else None}
    write_json(root / "results.json", {"campaigns": results, "summary": summary,
               "interpretation": "Independent per-target campaigns; incomplete denominators explicit"})
    fields = ["target_id", "seed", "condition", "status", "attack_success", "desired_probability", "desired_probability_delta",
              "target_flip_from_correct_clean", "attack_success_delta_vs_clean", "accuracy", "macro_f1", "duration_seconds", "updates", "tokens_seen"]
    with (root / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in results:
            flat = {k: r.get(k) for k in fields}
            if r["status"] == "complete":
                flat.update(desired_probability=r["target"]["probabilities"][r["desired_label"]],
                            accuracy=r["general"]["accuracy"], macro_f1=r["general"]["macro_f1"],
                            **{k: r["cost"][k] for k in ("duration_seconds", "updates", "tokens_seen")})
            writer.writerow(flat)
    return summary


def run(cfg, root, stage):
    bind_run(cfg, root)
    if stage in ("prepare", "run"):
        execute_stage(root, "prepare", lambda: prepare_stage(cfg, root))
    if stage in ("reference", "run"):
        execute_stage(root, "reference", lambda: reference_stage(cfg, root))
    if stage in ("poisons", "victims", "run"):
        targets = load_targets(root)
        for target in targets:
            target_key = target["id"].replace(":", "_")
            if stage in ("poisons", "run"):
                try:
                    execute_stage(root, "poison_"+target_key, lambda: poison_stage(cfg, root, target))
                except Exception as exc:
                    print(f"Poison campaign failed; retained in stage report: {exc}", flush=True)
                    if stage == "poisons":
                        evaluate(cfg, root)
                        raise
            if stage in ("victims", "run"):
                for seed in cfg["victim_seeds"]:
                    for condition in ("clean", "random", "optimized"):
                        try:
                            execute_stage(root, f"victim_{target_key}_{seed}_{condition}",
                                          lambda: victim_stage(cfg, root, target, seed, condition))
                        except Exception as exc:
                            print(f"Victim failed; retained in report: {exc}", flush=True)
    if stage in ("evaluate", "victims", "run"):
        summary = evaluate(cfg, root)
        print(summary, flush=True)
        if any(r["missing_or_failed"] for r in summary.values()):
            raise RuntimeError("Incomplete experiment; inspect stage failures/results.json")
