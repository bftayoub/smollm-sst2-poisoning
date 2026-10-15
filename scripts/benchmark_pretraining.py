"""Short throughput probe for randomly initialized SmolLM, not a training run.

Uses synthetic packed tokens and the standard full causal-LM loss. No model
weights are read or saved; only the existing architecture configuration is read.
"""
import argparse
import gc
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from poisonlab.common import write_json


def gpu_status():
    fields = "name,memory.total,memory.used,utilization.gpu,power.limit,power.draw,clocks.current.sm,temperature.gpu"
    return subprocess.check_output(["nvidia-smi", f"--query-gpu={fields}", "--format=csv"], text=True).strip()


def measure(config_dir, spec, seconds, min_updates, max_updates):
    torch.manual_seed(20261015)
    torch.cuda.manual_seed_all(20261015)
    torch.use_deterministic_algorithms(spec["precision"] == "fp32")
    torch.backends.cuda.matmul.allow_tf32 = spec["precision"] != "fp32"
    config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
    config.use_cache = False
    model = AutoModelForCausalLM.from_config(config, attn_implementation=spec["attention"],
                                            torch_dtype=torch.float32).to("cuda")
    model.train()
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95),
                                  weight_decay=0.1, fused=spec["precision"] != "fp32")
    batch = spec["microbatch"]
    length = spec["context"]
    accumulation = spec["accumulation"]
    # All positions are real input tokens; there is no padding to inflate rate.
    batches = [torch.randint(1, config.vocab_size, (batch, length), device="cuda") for _ in range(4)]
    counter = 0

    def update():
        nonlocal counter
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            ids = batches[counter % len(batches)]
            counter += 1
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=spec["precision"] == "bf16_amp"):
                loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            (loss/accumulation).backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, foreach=True)
        optimizer.step()
        torch.cuda.synchronize()
        value = float(loss.detach())
        if not torch.isfinite(norm) or not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss or gradient")
        return value

    before = gpu_status()
    for _ in range(3):
        last_loss = update()
    torch.cuda.reset_peak_memory_stats()
    times = []
    started = time.perf_counter()
    while len(times) < max_updates:
        t = time.perf_counter()
        last_loss = update()
        times.append(time.perf_counter()-t)
        if len(times) >= min_updates and time.perf_counter()-started >= seconds:
            break
    elapsed = time.perf_counter()-started
    tokens = batch*length*accumulation*len(times)
    result = {**spec, "status": "complete", "warmup_updates_excluded": 3,
              "measured_updates": len(times), "measured_input_tokens": tokens,
              "measured_supervised_positions": batch*(length-1)*accumulation*len(times),
              "duration_seconds": elapsed, "input_tokens_per_second": tokens/elapsed,
              "seconds_per_optimizer_update_median": statistics.median(times),
              "seconds_per_optimizer_update_min": min(times),
              "seconds_per_optimizer_update_max": max(times),
              "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
              "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved()/2**30,
              "parameter_count": sum(p.numel() for p in params),
              "master_parameter_dtype": str(params[0].dtype),
              "last_loss_synthetic_data": last_loss,
              "gpu_before": before, "gpu_after": gpu_status()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("runs/pilot-robust/initial"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/pretraining-feasibility"))
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--min-updates", type=int, default=8)
    parser.add_argument("--max-updates", type=int, default=80)
    parser.add_argument("--cases", nargs="+", help="Measure only these named cases")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This throughput probe requires CUDA")
    cases = [
        {"name": "fp32_eager_128", "precision": "fp32", "attention": "eager", "context": 128, "microbatch": 8, "accumulation": 1},
        {"name": "bf16_sdpa_512", "precision": "bf16_amp", "attention": "sdpa", "context": 512, "microbatch": 8, "accumulation": 1},
        {"name": "bf16_sdpa_1024", "precision": "bf16_amp", "attention": "sdpa", "context": 1024, "microbatch": 4, "accumulation": 1},
        {"name": "bf16_sdpa_2048_b2", "precision": "bf16_amp", "attention": "sdpa", "context": 2048, "microbatch": 2, "accumulation": 1},
        {"name": "bf16_sdpa_2048_b4", "precision": "bf16_amp", "attention": "sdpa", "context": 2048, "microbatch": 4, "accumulation": 1},
        {"name": "bf16_sdpa_2048_acc4", "precision": "bf16_amp", "attention": "sdpa", "context": 2048, "microbatch": 4, "accumulation": 4},
    ]
    if args.cases:
        unknown = set(args.cases)-{c["name"] for c in cases}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [c for c in cases if c["name"] in args.cases]
    report = {"technical_only": True, "synthetic_data": True, "random_initialization": True,
              "checkpoint_weights_saved": False,
              "caveat": "Short steady-state compute probe; excludes real data loading, tokenization, checkpoints and evaluation. Not a quality or convergence result.",
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "bf16_supported": torch.cuda.is_bf16_supported(),
              "initial_gpu_status": gpu_status(), "cases": []}
    for spec in cases:
        print(f"START {spec['name']}", flush=True)
        try:
            result = measure(args.config_dir, spec, args.seconds, args.min_updates, args.max_updates)
        except torch.cuda.OutOfMemoryError as exc:
            result = {**spec, "status": "out_of_memory", "error": str(exc).splitlines()[0]}
        report["cases"].append(result)
        write_json(args.output_dir / "throughput.json", report)
        print(json.dumps(result), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    report["status"] = "complete"
    write_json(args.output_dir / "throughput.json", report)


if __name__ == "__main__":
    main()
