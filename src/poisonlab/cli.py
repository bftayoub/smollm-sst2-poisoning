import argparse
from pathlib import Path

from .common import load_config
from .benchmark import benchmark
from .experiment import run
from .smoke import smoke


def main():
    parser = argparse.ArgumentParser(description="Targeted SST-2 synthetic poisoning research")
    parser.add_argument("stage", choices=["prepare", "reference", "poisons", "victims", "evaluate", "run", "smoke", "benchmark"])
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke(Path(args.run_dir or "runs/smoke"))
    else:
        cfg = load_config(args.config)
        root = Path(args.run_dir or "runs/"+cfg["run_name"])
        if args.stage == "benchmark":
            benchmark(cfg, root)
        else:
            run(cfg, root, args.stage)


if __name__ == "__main__":
    main()
