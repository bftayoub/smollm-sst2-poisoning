"""Real-model resource probe, not an attack experiment."""
import math
import time

import torch

from .common import Cost, device_for, seed_all, write_json
from .data import load_parts
from .model import load_model
from .poison import gradient, matching_step, parameters_for_alignment, safe_vocab


def benchmark(cfg, root):
    seed_all(cfg["reference_seed"], cfg["threads"])
    model, codec = load_model(cfg, root)
    parts = load_parts(root)
    device = device_for(cfg)
    cost = Cost(device)
    batch = codec.batch(parts["train"][:cfg["train"]["batch_size"]], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"])
    # Measure one full parameter update, then reload initial weights for GM probe.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t = time.monotonic()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model(**batch).loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = (time.monotonic()-t)/2
    del model, optimizer
    import gc
    gc.collect()
    model, codec = load_model(cfg, root)
    model.eval()
    _, params = parameters_for_alignment(model, cfg["poison"]["alignment_parameters"])
    row = {**parts["reserve"][0], "label": 1-parts["reserve"][0]["label"]}
    gt = tuple(g.detach() for g in gradient(model, params, codec.batch([row], device)))
    vocab = torch.tensor(safe_vocab(codec.tokenizer, cfg["poison"].get("vocab_limit")), device=device)
    z = torch.nn.Parameter(torch.randn(2, cfg["poison"]["length"], len(vocab), device=device)*0.01)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t = time.monotonic()
    value, dz = matching_step(model, codec, params, gt, z, vocab, [0, 1],
                              cfg["poison"]["temperature_start"], cfg["poison"]["microbatch"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    matching_seconds = time.monotonic()-t
    reference_updates = math.ceil(len(parts["train"])/cfg["train"]["batch_size"])*cfg["train"]["epochs"]
    victim_updates = math.ceil((len(parts["train"])+cfg["poison"]["count"])/cfg["train"]["batch_size"])*cfg["train"]["epochs"]
    result = {"technical_only": True, "checkpoint": "pretrained initial (not trained reference)",
              "training_updates_measured": 2, "training_seconds_per_update": training_seconds,
              "matching_poison_count_measured": 2, "matching_seconds_measured": matching_seconds,
              "matching_finite": bool(torch.isfinite(dz).all()), "matching_gradient_norm": float(dz.norm()),
              "alignment_loss": value, "device": str(device),
              "rough_reference_seconds": reference_updates*training_seconds,
              "rough_three_victim_seconds": 3*victim_updates*training_seconds,
              "rough_poison_optimization_seconds": matching_seconds*(cfg["poison"]["count"]/2)*cfg["poison"]["steps"],
              "estimate_caveat": "Two measured training batches only; lengths, allocation and token distributions vary. No efficacy result.",
              **cost.finish()}
    write_json(root / "resource_probe.json", result)
    print(result, flush=True)
