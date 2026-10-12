"""Export small auditable records; preserve all checkpoints separately under runs/."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for source in args.run_dir.rglob("*"):
        if source.suffix not in (".json", ".csv", ".log") or not source.is_file():
            continue
        relative = source.relative_to(args.run_dir)
        # HF checkpoint internals are available in the preserved local checkpoints.
        if relative.parts[0] == "initial" or source.name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json") and len(relative.parts) > 1:
            continue
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.name == "partitions.json":
            parts = json.loads(source.read_text())
            ids = {k: [{f: r[f] for f in ("id", "label", "normalized_hash")} for r in rows] for k, rows in parts.items()}
            target = target.with_name("partition_ids.json")
            target.write_text(json.dumps(ids, indent=2)+"\n")
        elif source.name == "targets.json":
            rows = json.loads(source.read_text())
            target.write_text(json.dumps([{k: v for k, v in r.items() if k != "text"} for r in rows], indent=2)+"\n")
        elif source.name == "target_selection.json":
            obj = json.loads(source.read_text())
            for row in obj.get("targets", []):
                row.pop("text", None)
            target.write_text(json.dumps(obj, indent=2)+"\n")
        elif source.name == "added_training_rows.json":
            # Keep synthetic text, omit resampled SST-2 text (source IDs suffice).
            rows = json.loads(source.read_text())
            for row in rows:
                if "source_id" in row:
                    row.pop("text", None)
            target.write_text(json.dumps(rows, indent=2)+"\n")
        else:
            shutil.copy2(source, target)


if __name__ == "__main__":
    main()
