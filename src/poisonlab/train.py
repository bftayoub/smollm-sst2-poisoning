import math
import random

import torch
from torch.utils.data import DataLoader

from .common import Cost, digest, rng_state, seed_all, write_json
from .data import assert_no_targets


@torch.no_grad()
def predict(model, codec, rows, batch_size=16):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    for start in range(0, len(rows), batch_size):
        group = rows[start:start+batch_size]
        batch = codec.batch(group, device, labeled=False)
        logits = model(**batch).logits
        last = batch["attention_mask"].sum(dim=1)-1
        binary = logits[torch.arange(len(group), device=device), last][:, codec.label_ids]
        probs = binary.softmax(dim=-1).cpu().tolist()
        raw = binary.cpu().tolist()
        for row, p, scores in zip(group, probs, raw):
            predictions.append({"id": row["id"], "label": row["label"],
                                "prediction": int(p[1] > p[0]), "probabilities": p,
                                "label_logits": scores})
    return predictions


def metrics(predictions):
    if not predictions:
        return {"accuracy": None, "macro_f1": None, "count": 0}
    f1 = []
    for label in (0, 1):
        tp = sum(r["label"] == label and r["prediction"] == label for r in predictions)
        fp = sum(r["label"] != label and r["prediction"] == label for r in predictions)
        fn = sum(r["label"] == label and r["prediction"] != label for r in predictions)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    return {"accuracy": sum(r["label"] == r["prediction"] for r in predictions)/len(predictions),
            "macro_f1": sum(f1)/2, "count": len(predictions)}


def train_model(model, codec, rows, cfg, seed, output, targets=()):
    assert_no_targets(rows, targets)
    seed_all(seed, cfg["threads"])
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    cost = Cost(device)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"],
                                  weight_decay=cfg["train"]["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(rows, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        generator=generator, collate_fn=lambda x: x)
    updates, tokens, extra_exposures, extra_tokens = 0, 0, 0, 0
    losses, orders = [], []
    for epoch in range(cfg["train"]["epochs"]):
        order = []
        for group in loader:
            batch = codec.batch(group, device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["max_grad_norm"])
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite training gradients")
            optimizer.step()
            updates += 1
            tokens += int(batch["attention_mask"].sum())
            for row, attention in zip(group, batch["attention_mask"]):
                order.append(row["id"])
                if row.get("added", False):
                    extra_exposures += 1
                    extra_tokens += int(attention.sum())
            losses.append(float(loss.detach()))
        orders.append(order)
        print(f"epoch={epoch+1} updates={updates} loss={losses[-1]:.5f}", flush=True)
    model.save_pretrained(output, safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "rng": rng_state(),
                "loader_generator": generator.get_state(), "updates": updates}, output / "training_state.pt")
    result = {**cost.finish(), "device": str(device), "seed": seed, "updates": updates, "tokens_seen": tokens,
              "added_example_exposures": extra_exposures, "added_prompt_label_tokens_seen": extra_tokens,
              "added_review_tokens_per_epoch": sum(len(codec.tokenizer.encode(r["text"], add_special_tokens=False))
                                                    for r in rows if r.get("added", False)),
              "row_count": len(rows), "last_loss": losses[-1], "training_rows_sha256": digest(rows),
              "training_config": cfg["train"]}
    write_json(output / "cost.json", result)
    write_json(output / "training_order.json", orders)
    write_json(output / "losses.json", losses)
    return result


def choose_targets(model, codec, reserve, cfg, root):
    preds = predict(model, codec, reserve, cfg["evaluation_batch_size"])
    write_json(root / "reserve_predictions.json", preds)
    eligible = []
    by_id = {r["id"]: r for r in reserve}
    for p in preds:
        if p["prediction"] == p["label"] and p["probabilities"][p["label"]] >= cfg["target_confidence"]:
            eligible.append({**by_id[p["id"]], "desired_label": 1-p["label"], "reference": p})
    eligible.sort(key=lambda r: (-r["reference"]["probabilities"][r["label"]], r["id"]))
    n = cfg["target_count"]
    if cfg["balanced_targets"]:
        if n % 2:
            raise ValueError("Balanced target count must be even")
        selected = [r for r in eligible if r["label"] == 0][:n//2] + [r for r in eligible if r["label"] == 1][:n//2]
    else:
        selected = eligible[:n]
    write_json(root / "target_selection.json", {"confidence_threshold": cfg["target_confidence"],
               "eligible_count": len(eligible), "requested_count": n, "selected_count": len(selected),
               "ambiguity_screen": "correct + binary confidence; no human semantic review",
               "targets": selected, "status": "complete" if len(selected) == n else "insufficient_targets"})
    if len(selected) != n:
        raise RuntimeError("Insufficient eligible targets; inspect development/reference, do not silently lower threshold")
    write_json(root / "targets.json", selected)
    write_json(root / "targets_manifest.json", {"targets_sha256": digest(selected),
               "selection_checkpoint": "reference", "reference_seed": cfg["reference_seed"],
               "confidence_threshold": cfg["target_confidence"]})
    return selected


def clean_additions(train, count, seed):
    rng = random.Random(seed)
    out = []
    for i in range(count):
        label = i % 2
        source = rng.choice([r for r in train if r["label"] == label])
        out.append({**source, "id": f"added:{i}", "source_id": source["id"], "added": True})
    return out
