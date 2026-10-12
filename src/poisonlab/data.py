import random
import re
import unicodedata

from .common import digest, read_json, write_json


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def rows_from(split, name):
    rows, seen = [], set()
    for i, row in enumerate(split):
        text = row["sentence"]
        key = normalize(text)
        if not key or key in seen:
            continue
        if int(row["label"]) not in (0, 1):
            raise ValueError("Official split contains unknown labels")
        seen.add(key)
        rows.append({"id": f"{name}:{i}", "text": text, "label": int(row["label"]),
                     "normalized_hash": digest(key)})
    return rows


def stratified(rows, count, rng):
    groups = [[r for r in rows if r["label"] == y] for y in (0, 1)]
    for group in groups:
        rng.shuffle(group)
    selected = groups[0][:count // 2] + groups[1][:count - count // 2]
    rng.shuffle(selected)
    ids = {r["id"] for r in selected}
    return selected, [r for r in rows if r["id"] not in ids]


def validate_partitions(parts):
    seen_ids, seen_text = set(), set()
    for name in ("train", "dev", "reserve", "general"):
        for row in parts[name]:
            key = normalize(row["text"])
            if row["id"] in seen_ids or key in seen_text:
                raise ValueError(f"Partition overlap at {name}/{row['id']}")
            seen_ids.add(row["id"])
            seen_text.add(key)


def partition(train, validation, cfg):
    rng = random.Random(cfg["split_seed"])
    train_rows = rows_from(train, "train")
    val_rows = rows_from(validation, "validation")
    # Official validation takes priority; remove ALL its normalized texts from train.
    val_keys = {normalize(r["text"]) for r in val_rows}
    train_rows = [r for r in train_rows if normalize(r["text"]) not in val_keys]
    selected, _ = stratified(train_rows, cfg["data"]["train_size"], rng)
    dev, rest = stratified(val_rows, cfg["data"]["dev_size"], rng)
    reserve, general = stratified(rest, cfg["data"]["reserve_size"], rng)
    parts = dict(train=selected, dev=dev, reserve=reserve, general=general)
    validate_partitions(parts)
    if any(not parts[k] for k in parts):
        raise ValueError("Empty partition; enlarge data or reduce requested sizes")
    return parts


def prepare(cfg, root):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    root.mkdir(parents=True, exist_ok=True)
    revision = HfApi().dataset_info(cfg["data"]["name"], revision=cfg["data"]["revision"]).sha
    ds = load_dataset(cfg["data"]["name"], revision=revision)
    parts = partition(ds["train"], ds["validation"], cfg)
    write_json(root / "partitions.json", parts)
    manifest = {"dataset": cfg["data"]["name"], "revision": revision,
                "split_seed": cfg["split_seed"], "requested": cfg["data"],
                "actual_sizes": {k: len(v) for k, v in parts.items()},
                "partition_sha256": digest(parts),
                "normalization": "NFKC + casefold + collapse whitespace",
                "official_test_used": False}
    write_json(root / "data_manifest.json", manifest)
    return manifest


def assert_no_targets(rows, targets):
    keys = {normalize(t["text"]) for t in targets}
    overlap = [r["id"] for r in rows if normalize(r["text"]) in keys]
    if overlap:
        raise ValueError(f"Target text injected into training: {overlap}")


def load_parts(root):
    parts = read_json(root / "partitions.json")
    validate_partitions(parts)
    if digest(parts) != read_json(root / "data_manifest.json")["partition_sha256"]:
        raise ValueError("Partition fingerprint changed")
    return parts
