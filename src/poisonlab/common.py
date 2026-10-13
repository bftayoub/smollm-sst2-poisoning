import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Resuming this checkpoint requires CUDA")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def device_for(cfg):
    value = cfg["device"]
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(value)


def environment():
    import datasets
    import transformers
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "datasets": datasets.__version__, "cpu_count": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


class Cost:
    def __init__(self, device):
        self.device = device
        self.start = time.monotonic()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def finish(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {"duration_seconds": time.monotonic() - self.start,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda" else None}


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    assert cfg["poison"]["count"] > 0 and cfg["poison"]["count"] % 2 == 0
    assert cfg["poison"]["steps"] > 0
    assert 0 < cfg["poison"]["temperature_end"] <= cfg["poison"]["temperature_start"]
    assert cfg["train"]["epochs"] > 0
    assert cfg["poison"]["microbatch"] > 0
    assert cfg["poison"]["length"] + 12 < cfg["max_length"]
    assert cfg["target_count"] > 0 and len(set(cfg["victim_seeds"])) == len(cfg["victim_seeds"])
    poison = cfg["poison"]
    assert poison.get("restarts", 1) >= 1
    assert poison.get("discrete_eval_every", 0) >= 0 and poison.get("discrete_samples", 0) >= 0
    assert poison.get("optimizer", "adam") in ("adam", "sign_adam")
    assert poison.get("initialization", "diffuse") in ("diffuse", "peaked_random")
    assert poison.get("target_objective", "label_ce") in ("label_ce", "binary_ce")
    assert poison.get("initialization_logit_gap", 15.0) > 0
    checkpoints = poison.get("alignment_checkpoints", ["reference"])
    assert "reference" in checkpoints and len(set(checkpoints)) == len(checkpoints)
    assert set(checkpoints) <= {"initial", "reference"}
    if "target_ids" in cfg:
        assert len(cfg["target_ids"]) == cfg["target_count"]
        assert len(set(cfg["target_ids"])) == cfg["target_count"]
    return cfg
