"""Resumable random-init LM training, optionally alternating with SST-2 batches."""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import yaml

from .common import (digest, environment, file_digest, read_json, restore_rng_state,
                     rng_state, write_json)
from .data import assert_no_targets, load_parts
from .lm_data import TokenBlocks, prepare_corpus, rounded_tokens, verify_corpus
from .model import LabelCodec
from .train import clean_additions, metrics, predict


def load_settings(path):
    cfg = yaml.safe_load(Path(path).read_text())
    data, train = cfg["data"], cfg["train"]
    for key in ("block_size", "shard_blocks", "shuffle_buffer", "validation_tokens"):
        if data[key] <= 0:
            raise ValueError(f"data.{key} must be positive")
    if not 0 < data["validation_buckets"] < 10000:
        raise ValueError("validation_buckets must be in (0,10000)")
    if not data["sources"] or any(not isinstance(s["weight"], int) or s["weight"] <= 0
                                  for s in data["sources"]):
        raise ValueError("Source weights must be positive integers")
    if len({s["name"] for s in data["sources"]}) != len(data["sources"]):
        raise ValueError("Source names must be unique")
    for key in ("micro_batch_size", "gradient_accumulation", "schedule_tokens", "until_tokens",
                "checkpoint_every_tokens", "validation_every_tokens", "validation_eval_tokens",
                "log_every_updates", "keep_last_checkpoints", "threads"):
        if train[key] <= 0:
            raise ValueError(f"train.{key} must be positive")
    if train["precision"] not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if not 0 <= train["warmup_tokens"] < train["schedule_tokens"]:
        raise ValueError("Warmup must be shorter than schedule")
    if train["learning_rate"] <= 0 or not 0 <= train["min_lr_ratio"] <= 1:
        raise ValueError("Invalid learning-rate settings")
    cls = cfg.get("classification", {})
    if cls.get("enabled") and (cls["every_lm_updates"] <= 0 or cls["batch_size"] <= 0
                               or cls["loss_weight"] <= 0):
        raise ValueError("Invalid classification cadence or loss weight")
    return cfg


@contextmanager
def exclusive_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def setup_seed(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # SDPA/BF16 fast path. Reproducible inputs and RNG; no promise of bitwise
    # equality across CUDA versions or hardware.
    torch.use_deterministic_algorithms(False)


def learning_rate(tokens, settings):
    warmup = settings["warmup_tokens"]
    peak = settings["learning_rate"]
    if tokens < warmup:
        return peak * tokens / max(1, warmup)
    fraction = min(1.0, (tokens - warmup) / (settings["schedule_tokens"] - warmup))
    ratio = settings["min_lr_ratio"]
    return peak * (ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * fraction)))


def classification_payload(cfg, tokenizer):
    spec = cfg.get("classification", {})
    if not spec.get("enabled"):
        return None
    source = Path(spec["source_run"])
    parts = load_parts(source)
    targets = read_json(source / "targets.json")
    manifest = read_json(source / "targets_manifest.json")
    if manifest["targets_sha256"] != digest(targets):
        raise ValueError("SST-2 target selection changed")
    poison_path = Path(spec["poison_file"])
    poison_rows = read_json(poison_path)
    poison_rows = [{**row, "added": True} for row in poison_rows]
    source_cfg = read_json(source / "config.json")
    count = source_cfg["poison"]["count"]
    if len(poison_rows) != count or count % 2:
        raise ValueError("Classification poison count differs from source protocol")
    if any(r.get("label") not in (0, 1) or not isinstance(r.get("text"), str) for r in poison_rows):
        raise ValueError("Expected labeled classification poisons, not arbitrary LM documents")
    expected_labels = [i % 2 for i in range(count)]
    if [r["label"] for r in poison_rows] != expected_labels:
        raise ValueError("Poison labels must match the alternating clean-addition order")
    if len({r["id"] for r in parts["train"] + poison_rows}) != len(parts["train"]) + count:
        raise ValueError("Duplicate SST-2 row IDs")
    assert_no_targets(parts["train"] + poison_rows, parts["reserve"])
    codec = LabelCodec(tokenizer, spec["max_length"])
    expected = read_json(source / "model_manifest.json")["labels"]
    if codec.label_ids != expected["label_ids"] or codec.codes != expected["codes"]:
        raise ValueError("Classification tokenizer/label codes differ from source experiment")
    clean = clean_additions(parts["train"], count, source_cfg["poison_seed"])
    return {"parts": parts, "targets": targets, "codec": codec,
            "rows": {"clean": parts["train"] + clean, "poisoned": parts["train"] + poison_rows},
            "identity": {"partition_sha256": digest(parts), "target_sha256": digest(targets),
                         "poison_file_sha256": file_digest(poison_path),
                         "clean_additions_sha256": digest(clean), "labels": codec.metadata(),
                         "provenance": spec["provenance"]}}


def classification_batch(rows, update, batch_size, seed, cache):
    """Same shuffled row indices in the two conditions; cursor derives from update."""
    group = []
    for position in range(update * batch_size, (update + 1) * batch_size):
        epoch, offset = divmod(position, len(rows))
        if cache.get("epoch") != epoch:
            cache["epoch"] = epoch
            cache["order"] = np.random.default_rng(seed + epoch).permutation(len(rows))
        group.append(rows[int(cache["order"][offset])])
    return group


def ensure_initial(cfg, data_root, run_root, identity):
    path = run_root / "initial"
    meta = run_root / "initial_manifest.json"
    if meta.exists():
        saved = read_json(meta)
        if saved["identity"] != identity or file_digest(path / "model.safetensors") != saved["weights_sha256"]:
            raise ValueError("Random initial checkpoint changed")
        return saved
    if path.exists():
        raise ValueError("Incomplete initial checkpoint; inspect and remove it before retrying")
    setup_seed(cfg["train"]["seed"], cfg["train"]["threads"])
    config = AutoConfig.from_pretrained(data_root / "metadata", local_files_only=True)
    # Critical: config only, no pretrained model weights.
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager",
                                            torch_dtype=torch.float32)
    tmp = run_root / "initial.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp, safe_serialization=True)
    AutoTokenizer.from_pretrained(data_root / "metadata", local_files_only=True).save_pretrained(tmp)
    tmp.replace(path)
    result = {"identity": identity, "initialization": "random_from_architecture_config",
              "seed": cfg["train"]["seed"], "parameter_count": sum(p.numel() for p in model.parameters()),
              "weights_sha256": file_digest(path / "model.safetensors")}
    write_json(meta, result)
    del model
    return result


@contextmanager
def interruption_flag():
    flag = {"requested": False}
    previous = {}
    def request(signum, frame):
        flag["requested"] = True
        print("Stop requested; saving after the current complete update.", flush=True)
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, request)
    try:
        yield flag
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def append_log(path, event):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(event, allow_nan=False) + "\n")


@torch.no_grad()
def validation_loss(model, blocks, batch_size, amp, device, token_limit):
    model.eval()
    count = min(len(blocks), max(1, math.ceil(token_limit / blocks.length)))
    total = 0.0
    for start in range(0, count, batch_size):
        n = min(batch_size, count - start)
        ids = torch.from_numpy(blocks.batch(start, n)).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")
        total += float(loss) * n
    model.train()
    value = total / count
    return {"loss": value, "perplexity": math.exp(min(value, 80)),
            "input_tokens": count * blocks.length}


def evaluate_model(model, validation, payload, cfg, amp, device, full=False):
    settings = cfg["train"]
    limit = len(validation) * validation.length if full else settings["validation_eval_tokens"]
    result = {"language_model": validation_loss(model, validation, settings["micro_batch_size"],
                                                amp, device, limit)}
    if payload:
        batch = cfg["classification"]["evaluation_batch_size"]
        general = predict(model, payload["codec"], payload["parts"]["general"], batch)
        target = predict(model, payload["codec"], payload["targets"], batch)
        result.update(classification=metrics(general), targets=target)
    model.train()
    return result


def save_checkpoint(model, optimizer, tokenizer, condition_root, state, cfg):
    parent = condition_root / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    name = f"step-{state['lm_updates']:09d}"
    target = parent / name
    # A final evaluation may revisit an already saved step. Its training state
    # is unchanged, so the existing atomic checkpoint remains authoritative.
    if not target.exists():
        tmp = parent / (name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        model.save_pretrained(tmp, safe_serialization=True)
        tokenizer.save_pretrained(tmp)
        torch.save({**state, "optimizer": optimizer.state_dict(), "rng": rng_state()},
                   tmp / "training_state.pt")
        write_json(tmp / "metadata.json", {k: v for k, v in state.items() if k != "rng"})
        write_json(tmp / "checksums.json", {p.name: file_digest(p) for p in
                   (tmp / "model.safetensors", tmp / "training_state.pt")})
        tmp.replace(target)
    write_json(condition_root / "latest.json", {"checkpoint": str(Path("checkpoints") / name),
               "tokens": state["tokens"], "lm_updates": state["lm_updates"]})
    checkpoints = sorted(p for p in parent.glob("step-*") if p.is_dir() and not p.name.endswith(".tmp"))
    recent = set(checkpoints[-cfg["train"]["keep_last_checkpoints"]:])
    protected_steps = {rounded_tokens(t, cfg) // state["tokens_per_update"]
                       for t in cfg["train"]["milestones"]}
    for path in checkpoints:
        if path not in recent and int(path.name.split("-")[1]) not in protected_steps:
            shutil.rmtree(path)
    return target


def train_condition(cfg, data_root, run_root, condition, until_tokens, manifest, payload, initial):
    settings = cfg["train"]
    device = torch.device(settings["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp = settings["precision"] == "bf16"
    if amp and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 configuration requires a compatible CUDA GPU")
    if condition == "poisoned" and not payload:
        raise ValueError("Poisoned condition requires enabled, labeled SST-2 poisoning")
    per_update = rounded_tokens(1, cfg)
    target = rounded_tokens(until_tokens, cfg)
    if until_tokens <= 0 or until_tokens > settings["schedule_tokens"]:
        raise ValueError("until-tokens must be positive and at most schedule_tokens")
    if manifest["train_tokens"] < target:
        raise ValueError(f"Prepared corpus too small ({manifest['train_tokens']}); run prepare --until-tokens {until_tokens}")
    if any(s["validation_tokens"] != s["validation_quota"] for s in manifest["sources"]):
        raise ValueError("Corpus preparation is incomplete: validation quota not reached")
    length = cfg["data"]["block_size"]
    blocks = TokenBlocks(data_root, manifest["parts"], length)
    validation = TokenBlocks(data_root, manifest["validation"], length)
    if not len(validation):
        raise ValueError("Empty validation corpus")
    output = run_root / condition
    output.mkdir(parents=True, exist_ok=True)
    setup_seed(settings["seed"], settings["threads"])
    checkpoint = run_root / "initial"
    saved = None
    if (output / "latest.json").exists():
        checkpoint = output / read_json(output / "latest.json")["checkpoint"]
        for filename, expected in read_json(checkpoint / "checksums.json").items():
            if file_digest(checkpoint / filename) != expected:
                raise ValueError("Checkpoint fingerprint changed")
        saved = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        old_parts = saved["data_parts"]
        if manifest["parts"][:len(old_parts)] != old_parts:
            raise ValueError("Corpus prefix changed since checkpoint; continuation refused")
        if saved["initial_weights_sha256"] != initial["weights_sha256"]:
            raise ValueError("Initial checkpoint differs from saved run")
        if saved["tokens"] >= target:
            cost_path, eval_path = output / "result.json", output / "evaluation.json"
            if (cost_path.exists() and read_json(cost_path)["status"] == "budget_reached"
                    and eval_path.exists() and read_json(eval_path)["tokens"] == saved["tokens"]):
                print(f"{condition}: already at {saved['tokens']:,} tokens", flush=True)
                return "already_complete"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
                  torch_dtype=torch.float32, attn_implementation=settings["attention"]).to(device)
    model.config.use_cache = False
    model.train()
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"],
            betas=tuple(settings["betas"]), weight_decay=settings["weight_decay"], fused=device.type == "cuda")
    tokenizer = AutoTokenizer.from_pretrained(data_root / "metadata", local_files_only=True)
    state = {"lm_updates": 0, "classification_updates": 0, "tokens": 0,
             "classification_tokens": 0, "poison_exposures": 0, "added_exposures": 0,
             "wall_seconds": 0.0, "condition": condition, "tokens_per_update": per_update,
             "initial_weights_sha256": initial["weights_sha256"], "data_parts": manifest["parts"]}
    if saved:
        optimizer.load_state_dict(saved.pop("optimizer"))
        restore_rng_state(saved.pop("rng"))
        state.update(saved)
        state["data_parts"] = manifest["parts"]
        del saved
    # Remove logs after the last authoritative checkpoint on resumption.
    log = output / "training.jsonl"
    if log.exists():
        events = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        log.write_text("".join(json.dumps(e) + "\n" for e in events if e["lm_updates"] <= state["lm_updates"]))
    sampler_cache = {}
    started = time.monotonic()
    prior_wall = state["wall_seconds"]
    start_tokens = state["tokens"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    milestones = {rounded_tokens(t, cfg) for t in settings["milestones"]}
    last_checkpoint_tokens = state["tokens"]
    last_validation_tokens = state["tokens"]
    status = "budget_reached"

    def record_evaluation(full=False):
        result = evaluate_model(model, validation, payload, cfg, amp, device, full)
        event = {"event": "evaluation", "lm_updates": state["lm_updates"], "tokens": state["tokens"], **result}
        append_log(log, event)
        write_json(output / "evaluation.json", event)
        print(f"{condition} eval tokens={state['tokens']:,} lm_loss={result['language_model']['loss']:.4f}" +
              (f" sst2_accuracy={result['classification']['accuracy']:.4f}" if payload else ""), flush=True)

    with interruption_flag() as stop:
        if state["lm_updates"] == 0:
            record_evaluation()
        while state["tokens"] < target:
            next_tokens = state["tokens"] + per_update
            lr = learning_rate(min(next_tokens, settings["schedule_tokens"]), settings)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            cursor = state["tokens"] // length
            for micro in range(settings["gradient_accumulation"]):
                ids = torch.from_numpy(blocks.batch(cursor + micro * settings["micro_batch_size"],
                                                    settings["micro_batch_size"])).to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite LM training loss")
                (loss / settings["gradient_accumulation"]).backward()
                loss_sum += float(loss.detach()) / settings["gradient_accumulation"]
            norm = torch.nn.utils.clip_grad_norm_(parameters, settings["max_grad_norm"], foreach=True)
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite LM gradients")
            optimizer.step()
            state["lm_updates"] += 1
            state["tokens"] = next_tokens
            cls_loss = None
            if payload and state["lm_updates"] % cfg["classification"]["every_lm_updates"] == 0:
                spec = cfg["classification"]
                rows = classification_batch(payload["rows"][condition], state["classification_updates"],
                                            spec["batch_size"], settings["seed"], sampler_cache)
                batch = payload["codec"].batch(rows, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    raw_loss = model(**batch).loss
                    loss = raw_loss * spec["loss_weight"]
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite classification loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, settings["max_grad_norm"], foreach=True)
                if not torch.isfinite(norm):
                    raise RuntimeError("Non-finite classification gradients")
                optimizer.step()
                cls_loss = float(raw_loss.detach())
                state["classification_updates"] += 1
                state["classification_tokens"] += int(batch["attention_mask"].sum())
                added = sum(bool(r.get("added")) for r in rows)
                state["added_exposures"] += added
                state["poison_exposures"] += added if condition == "poisoned" else 0
            state["wall_seconds"] = prior_wall + time.monotonic() - started
            if state["lm_updates"] % settings["log_every_updates"] == 0:
                elapsed = max(time.monotonic() - started, 1e-6)
                event = {"event": "train", "lm_updates": state["lm_updates"], "tokens": state["tokens"],
                         "lm_loss": loss_sum, "classification_loss": cls_loss, "learning_rate": lr,
                         "tokens_per_second_this_invocation": (state["tokens"] - start_tokens) / elapsed,
                         "classification_updates": state["classification_updates"],
                         "added_exposures": state["added_exposures"], "poison_exposures": state["poison_exposures"]}
                append_log(log, event)
                print(f"{condition} tokens={state['tokens']:,} lm_loss={loss_sum:.4f} lr={lr:.6g} "
                      f"tokens/s={event['tokens_per_second_this_invocation']:.0f}", flush=True)
            at_end = state["tokens"] >= target
            milestone = state["tokens"] in milestones
            if (not stop["requested"] and (at_end or milestone or
                state["tokens"] - last_validation_tokens >= settings["validation_every_tokens"])):
                record_evaluation(full=at_end)
                last_validation_tokens = state["tokens"]
            state["wall_seconds"] = prior_wall + time.monotonic() - started
            if (at_end or milestone or stop["requested"] or
                    state["tokens"] - last_checkpoint_tokens >= settings["checkpoint_every_tokens"]):
                save_checkpoint(model, optimizer, tokenizer, output, state, cfg)
                last_checkpoint_tokens = state["tokens"]
            if stop["requested"]:
                status = "interrupted"
                break
        # An interrupt can arrive at the exact token limit before final eval.
        # On the next launch, finalize that checkpoint without another update.
        if state["tokens"] >= target and not stop["requested"]:
            eval_path = output / "evaluation.json"
            if not eval_path.exists() or read_json(eval_path)["tokens"] != state["tokens"]:
                record_evaluation(full=True)
            state["wall_seconds"] = prior_wall + time.monotonic() - started
            save_checkpoint(model, optimizer, tokenizer, output, state, cfg)
    cost = {**state, "status": status, "requested_until_tokens": until_tokens,
            "rounded_until_tokens": target,
            "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
            "latest_checkpoint": read_json(output / "latest.json")["checkpoint"]}
    write_json(output / "result.json", cost)
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return status


def pair_summary(root, payload):
    result = {"conditions": {}, "interpretation": "Alternating LM/SST-2 from random weights; exploratory classification poisoning"}
    for condition in ("clean", "poisoned"):
        path = root / condition / "result.json"
        if path.exists():
            result["conditions"][condition] = read_json(path)
    a, b = result["conditions"].get("clean"), result["conditions"].get("poisoned")
    if payload and a and b and a["tokens"] == b["tokens"] and a["status"] == b["status"] == "budget_reached":
        if a["initial_weights_sha256"] != b["initial_weights_sha256"] or a["classification_updates"] != b["classification_updates"]:
            raise ValueError("Unpaired initial weights or classification budgets")
        clean = read_json(root / "clean" / "evaluation.json")
        poisoned = read_json(root / "poisoned" / "evaluation.json")
        if clean["tokens"] == poisoned["tokens"] == a["tokens"]:
            by_id = {r["id"]: r for r in clean["targets"]}
            attacks = []
            for row, target in zip(poisoned["targets"], payload["targets"]):
                baseline = by_id[row["id"]]
                desired = target["desired_label"]
                attacks.append({"id": row["id"], "desired_label": desired,
                    "clean_prediction": baseline["prediction"], "poisoned_prediction": row["prediction"],
                    "clean_correct": baseline["prediction"] == target["label"],
                    "flip_from_correct_clean": baseline["prediction"] == target["label"] and row["prediction"] == desired,
                    "desired_probability_delta": row["probabilities"][desired] - baseline["probabilities"][desired]})
            result.update(paired=True, targets=attacks, clean_classification=clean["classification"],
                          poisoned_classification=poisoned["classification"])
    write_json(root / "results.json", result)
    return result


def run_training(cfg, data_root, root, conditions, until_tokens):
    data_root, root = Path(data_root).resolve(), Path(root).resolve()
    with exclusive_lock(root):
        # Training holds a shared data lock so preparation cannot append midway.
        with (data_root / ".lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Corpus preparation is still running") from exc
            manifest = verify_corpus(data_root)
            identity = read_json(data_root / "identity.json")
            if identity["data_config"] != cfg["data"]:
                raise ValueError("Training configuration differs from prepared corpus")
            if identity["model_spec"] != cfg["model"]:
                raise ValueError("Training model/tokenizer differs from prepared corpus")
            tokenizer = AutoTokenizer.from_pretrained(data_root / "metadata", local_files_only=True)
            payload = classification_payload(cfg, tokenizer)
            bound = {"config": cfg, "data_root": str(data_root), "corpus_identity": digest(identity),
                     "classification": payload["identity"] if payload else None}
            bind_path = root / "run_manifest.json"
            if bind_path.exists() and read_json(bind_path) != bound:
                raise ValueError("Run configuration or fixed poisons changed; use a new run directory")
            write_json(bind_path, bound)
            write_json(root / "environment.json", environment())
            initial = ensure_initial(cfg, data_root, root, digest(bound))
            for condition in conditions:
                status = train_condition(cfg, data_root, root, condition, until_tokens, manifest, payload, initial)
                pair_summary(root, payload)
                if status == "interrupted":
                    raise SystemExit(130)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "status"])
    parser.add_argument("--config", default="configs/from-scratch.yaml")
    parser.add_argument("--data-dir", default="runs/from-scratch-corpus")
    parser.add_argument("--run-dir", default="runs/from-scratch-classification")
    parser.add_argument("--until-tokens", type=int, help="LM input-token budget; rounded up to a complete update")
    parser.add_argument("--conditions", nargs="+", choices=["clean", "poisoned"])
    args = parser.parse_args()
    cfg = load_settings(args.config)
    tokens = args.until_tokens if args.until_tokens is not None else cfg["train"]["until_tokens"]
    if tokens <= 0 or tokens > cfg["train"]["schedule_tokens"]:
        parser.error("until-tokens must be positive and no larger than schedule_tokens")
    if args.stage == "prepare":
        with exclusive_lock(args.data_dir):
            prepare_corpus(cfg, Path(args.data_dir), tokens)
    elif args.stage == "status":
        path = Path(args.run_dir)
        for condition in ("clean", "poisoned"):
            result = path / condition / "latest.json"
            print(condition, json.dumps(read_json(result)) if result.exists() else "not started")
    else:
        conditions = args.conditions or (["clean", "poisoned"] if cfg.get("classification", {}).get("enabled") else ["clean"])
        if len(set(conditions)) != len(conditions):
            parser.error("Duplicate conditions")
        run_training(cfg, args.data_dir, args.run_dir, conditions, tokens)


if __name__ == "__main__":
    main()
