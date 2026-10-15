"""Appendable, immutable token shards with a document-level held-out split."""
import bisect
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from transformers import AutoConfig, AutoTokenizer

from .common import digest, file_digest, read_json, write_json


def tokenizer_digest(tokenizer):
    # Includes tokenization rules and special-token definitions, not local paths.
    if not tokenizer.is_fast:
        raise ValueError("Corpus preparation requires a fast tokenizer")
    return digest({"backend": json.loads(tokenizer.backend_tokenizer.to_str()),
                   "special_tokens": tokenizer.special_tokens_map})


def architecture_digest(config):
    values = config.to_dict()
    return digest({key: value for key, value in values.items()
                   if not key.startswith("_") and key not in ("transformers_version", "torch_dtype")})


def rounded_tokens(tokens, cfg):
    per_update = (cfg["data"]["block_size"] * cfg["train"]["micro_batch_size"] *
                  cfg["train"]["gradient_accumulation"])
    return math.ceil(tokens / per_update) * per_update


def document_key(text):
    # Preserve the original training text. Whitespace normalization is only
    # used to assign duplicates to the same split and to exclude exact repeats.
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def source_rows(spec, data, revision, consumed):
    if "path" in spec:
        def local_rows():
            with Path(spec["path"]).open() as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
        return itertools.islice(local_rows(), consumed, None)
    from datasets import load_dataset
    stream = load_dataset(data["dataset"], spec["name"], revision=revision,
                          split="train", streaming=True)
    stream = stream.shuffle(seed=data["seed"], buffer_size=data["shuffle_buffer"])
    # Replays the deterministic stream without retokenizing consumed documents.
    return iter(stream.skip(consumed))


def _metadata(cfg, root):
    model_spec, data = cfg["model"], cfg["data"]
    revision = data["revision"]
    if not all("path" in s for s in data["sources"]):
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info(data["dataset"], revision=revision).sha
    source = model_spec.get("local_config") or model_spec["name"]
    kwargs = {"local_files_only": True} if model_spec.get("local_config") else {
        "revision": model_spec["revision"]}
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    config = AutoConfig.from_pretrained(source, **kwargs)
    config.use_cache = False
    if config.vocab_size > 65536 or len(tokenizer) > 65536:
        raise ValueError("uint16 token shards require a vocabulary of at most 65536 IDs")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    if data["block_size"] > config.max_position_embeddings:
        raise ValueError("block_size exceeds model context")
    sources = [{**s, **({"file_sha256": file_digest(s["path"])} if "path" in s else {})}
               for s in data["sources"]]
    identity = {"data_config": data, "model_spec": model_spec,
                "resolved_revision": revision, "sources": sources,
                "model_config_sha256": architecture_digest(config),
                "tokenizer_sha256": tokenizer_digest(tokenizer),
                "split_rule": "normalized document SHA256 modulo 10000, heldout first buckets",
                "format": "little-endian uint16; packed blocks with document EOS"}
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(root / "metadata")
    config.save_pretrained(root / "metadata")
    write_json(root / "identity.json", identity)
    return identity, tokenizer


def prepare_corpus(cfg, root, tokens):
    """Prepare/extend the same stream; interrupted writes roll back to manifest."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data = cfg["data"]
    length = data["block_size"]
    target = rounded_tokens(tokens, cfg)
    specs = data["sources"]
    cycle = [i for i, s in enumerate(specs) for _ in range(s["weight"])]
    total_weight = len(cycle)
    if (root / "identity.json").exists():
        identity = read_json(root / "identity.json")
        if identity["data_config"] != data:
            raise ValueError("Corpus configuration changed; use a new data directory")
        if identity["model_spec"] != cfg["model"]:
            raise ValueError("Corpus model/tokenizer configuration changed")
        for spec in identity["sources"]:
            if "path" in spec and file_digest(spec["path"]) != spec["file_sha256"]:
                raise ValueError("Local source changed")
        tokenizer = AutoTokenizer.from_pretrained(root / "metadata", local_files_only=True)
        if tokenizer_digest(tokenizer) != identity["tokenizer_sha256"]:
            raise ValueError("Tokenizer changed")
    else:
        identity, tokenizer = _metadata(cfg, root)
    state_path = root / "manifest.json"
    if state_path.exists():
        state = read_json(state_path)
        verify_corpus(root, state)
    else:
        validation_blocks = math.ceil(data["validation_tokens"] / length)
        quotas = [sum(cycle[j % total_weight] == i for j in range(validation_blocks)) * length
                  for i in range(len(specs))]
        state = {"identity_sha256": digest(identity), "block_size": length,
                 "train_tokens": 0, "parts": [], "ledger_bytes": 0,
                 "sources": [{"consumed": 0, "train_buffer": [], "validation_tokens": 0,
                              "validation_quota": quota} for quota in quotas],
                 "validation": []}
        write_json(state_path, state)
    if state["identity_sha256"] != digest(identity):
        raise ValueError("Corpus identity changed")
    ledger_path = root / "documents.jsonl"
    with ledger_path.open("a+b") as ledger:
        ledger.truncate(state["ledger_bytes"])
    seen = set()
    with ledger_path.open() as ledger:
        for line in ledger:
            seen.add(json.loads(line)["sha256"])
    validation_handles = []
    for i, item in enumerate(state["sources"]):
        path = root / f"validation-{i:02d}.bin"
        with path.open("a+b") as handle:
            handle.truncate(item["validation_tokens"] * 2)
        validation_handles.append(path.open("ab"))
    # Construct streams lazily: rerunning a completed preparation needs no network.
    streams = {}
    ledger = ledger_path.open("a")

    def consume(i, keep_train=True):
        spec, item = specs[i], state["sources"][i]
        if i not in streams:
            streams[i] = source_rows(spec, data, identity["resolved_revision"], item["consumed"])
        try:
            row = next(streams[i])
        except StopIteration as exc:
            raise RuntimeError(f"Source {spec['name']} exhausted before token budget") from exc
        item["consumed"] += 1
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            return
        key = document_key(text)
        if key in seen:
            return
        seen.add(key)
        is_validation = int(key[:16], 16) % 10000 < data["validation_buckets"]
        ledger.write(json.dumps({"sha256": key, "source": i,
                                 "split": "validation" if is_validation else "train"}) + "\n")
        if is_validation and item["validation_tokens"] < item["validation_quota"]:
            ids = tokenizer.encode(text, add_special_tokens=False, verbose=False) + [tokenizer.eos_token_id]
            ids = ids[:item["validation_quota"] - item["validation_tokens"]]
            validation_handles[i].write(np.asarray(ids, dtype="<u2").tobytes())
            item["validation_tokens"] += len(ids)
        elif not is_validation and keep_train:
            ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            item["train_buffer"].extend(ids + [tokenizer.eos_token_id])

    def commit():
        ledger.flush()
        state["ledger_bytes"] = ledger_path.stat().st_size
        state["validation"] = []
        for i, handle in enumerate(validation_handles):
            handle.flush()
            path = root / f"validation-{i:02d}.bin"
            state["validation"].append({"file": path.name,
                "tokens": state["sources"][i]["validation_tokens"], "sha256": file_digest(path)})
        write_json(state_path, state)
        print(f"corpus train_tokens={state['train_tokens']:,} validation_tokens="
              f"{sum(s['validation_tokens'] for s in state['sources']):,}", flush=True)

    try:
        chunk_blocks = data.get("shard_blocks", 4096)
        while state["train_tokens"] < target:
            blocks = []
            count = min(chunk_blocks, (target - state["train_tokens"]) // length)
            for offset in range(count):
                block_index = state["train_tokens"] // length + offset
                i = cycle[block_index % total_weight]
                item = state["sources"][i]
                while len(item["train_buffer"]) < length:
                    consume(i)
                blocks.extend(item["train_buffer"][:length])
                del item["train_buffer"][:length]
            name = f"train-{len(state['parts']):06d}.bin"
            path = root / name
            tmp = path.with_suffix(".tmp")
            np.asarray(blocks, dtype="<u2").tofile(tmp)
            tmp.replace(path)
            state["parts"].append({"file": name, "tokens": len(blocks), "sha256": file_digest(path)})
            state["train_tokens"] += len(blocks)
            commit()
        for i, item in enumerate(state["sources"]):
            while item["validation_tokens"] < item["validation_quota"]:
                consume(i, keep_train=False)
                if item["consumed"] % 10000 == 0:
                    commit()
        commit()
    finally:
        # On interruption, manifest remains the last committed source position;
        # uncommitted ledger/validation tails are truncated on the next call.
        ledger.close()
        for handle in validation_handles:
            handle.close()
    return state


def verify_corpus(root, manifest=None):
    root = Path(root)
    manifest = manifest or read_json(root / "manifest.json")
    identity = read_json(root / "identity.json")
    if manifest["identity_sha256"] != digest(identity):
        raise ValueError("Corpus identity fingerprint mismatch")
    config = AutoConfig.from_pretrained(root / "metadata", local_files_only=True)
    if architecture_digest(config) != identity["model_config_sha256"]:
        raise ValueError("Corpus architecture changed")
    if sum(p["tokens"] for p in manifest["parts"]) != manifest["train_tokens"]:
        raise ValueError("Corpus token total mismatch")
    if any(p["tokens"] <= 0 or p["tokens"] % manifest["block_size"] for p in manifest["parts"]):
        raise ValueError("Training shards must contain complete blocks")
    for part in manifest["parts"] + manifest["validation"]:
        path = root / part["file"]
        # Preparation can have an uncommitted validation tail after interruption.
        if part["file"].startswith("validation-"):
            if path.stat().st_size < part["tokens"] * 2:
                raise ValueError(f"Truncated validation shard: {path}")
            with path.open("rb") as handle:
                value = hashlib.sha256(handle.read(part["tokens"] * 2)).hexdigest()
        else:
            if path.stat().st_size != part["tokens"] * 2:
                raise ValueError(f"Token shard size changed: {path}")
            value = file_digest(path)
        if value != part["sha256"]:
            raise ValueError(f"Token shard fingerprint changed: {path}")
    return manifest


class TokenBlocks:
    def __init__(self, root, parts, length):
        self.length = length
        self.arrays, self.ends = [], []
        total = 0
        for item in parts:
            blocks = item["tokens"] // length
            if not blocks:
                continue
            self.arrays.append(np.memmap(Path(root) / item["file"], mode="r", dtype="<u2",
                                         shape=(blocks, length)))
            total += blocks
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def batch(self, start, count):
        if start < 0 or start + count > len(self):
            raise IndexError("Token budget exceeds prepared corpus; extend preparation first")
        rows = []
        for index in range(start, start + count):
            shard = bisect.bisect_right(self.ends, index)
            offset = self.ends[shard - 1] if shard else 0
            rows.append(self.arrays[shard][index - offset])
        return np.asarray(rows, dtype=np.int64)
