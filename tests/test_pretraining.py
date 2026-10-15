import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml
from transformers import AutoModelForCausalLM

from poisonlab.common import digest, read_json, write_json
from poisonlab.lm_data import (TokenBlocks, document_key, prepare_corpus,
                              rounded_tokens, verify_corpus)
from poisonlab.model import LabelCodec, prepare_initial
from poisonlab.pretrain import learning_rate, load_settings, run_training
from poisonlab.smoke import tiny_components


@pytest.fixture
def setup(tmp_path):
    model, tokenizer = tiny_components()
    metadata = tmp_path / "architecture"
    model.config.max_position_embeddings = 64
    model.save_pretrained(metadata)
    tokenizer.save_pretrained(metadata)
    rows = [{"text": f"good movie {i} " + "good bad movie " * 8} for i in range(400)]
    for source in ("web", "synthetic"):
        path = tmp_path / (source + ".jsonl")
        path.write_text("".join(json.dumps({"text": source + " " + r["text"]}) + "\n" for r in rows))
    cfg = {"run_name": "tiny-pretraining", "model": {"name": "unused", "revision": "local",
              "local_config": str(metadata)},
        "data": {"dataset": "local-fixture", "revision": "local", "sources": [
            {"name": "web", "path": str(tmp_path / "web.jsonl"), "weight": 4},
            {"name": "synthetic", "path": str(tmp_path / "synthetic.jsonl"), "weight": 1}],
            "seed": 9, "shuffle_buffer": 8, "block_size": 16, "shard_blocks": 4,
            "validation_tokens": 80, "validation_buckets": 2000},
        "train": {"device": "cpu", "threads": 1, "seed": 11, "precision": "fp32",
            "attention": "eager", "micro_batch_size": 2, "gradient_accumulation": 2,
            "until_tokens": 128, "schedule_tokens": 512, "warmup_tokens": 64,
            "learning_rate": 0.001, "min_lr_ratio": 0.1, "weight_decay": 0.1,
            "betas": [0.9, 0.95], "max_grad_norm": 1.0, "checkpoint_every_tokens": 64,
            "validation_every_tokens": 128, "validation_eval_tokens": 32,
            "milestones": [128, 256, 512], "log_every_updates": 1, "keep_last_checkpoints": 2},
        "classification": {"enabled": False}}
    return cfg, tmp_path / "corpus", tmp_path


def latest_weights(root, condition="clean"):
    latest = read_json(root / condition / "latest.json")
    model = AutoModelForCausalLM.from_pretrained(root / condition / latest["checkpoint"],
                                                local_files_only=True, attn_implementation="eager")
    return {key: value.clone() for key, value in model.state_dict().items()}


def add_classification(cfg, tmp_path):
    source = tmp_path / "sst2-source"
    _, tokenizer = tiny_components()
    codec = LabelCodec(tokenizer, 48)
    parts = {name: [{"id": f"{name}:{i}", "text": f"{name} movie {i}", "label": i % 2}
                    for i in range(count)]
             for name, count in [("train", 8), ("dev", 2), ("reserve", 2), ("general", 4)]}
    write_json(source / "partitions.json", parts)
    write_json(source / "data_manifest.json", {"partition_sha256": digest(parts)})
    targets = [{**parts["reserve"][0], "desired_label": 1}]
    write_json(source / "targets.json", targets)
    write_json(source / "targets_manifest.json", {"targets_sha256": digest(targets)})
    write_json(source / "model_manifest.json", {"labels": codec.metadata()})
    write_json(source / "config.json", {"poison": {"count": 2}, "poison_seed": 123})
    poison = source / "poisons.json"
    write_json(poison, [{"id": f"added:{i}", "text": "bad good " * (i + 1),
                         "label": i, "added": True} for i in range(2)])
    cfg["classification"] = {"enabled": True, "source_run": str(source),
        "poison_file": str(poison), "every_lm_updates": 1, "batch_size": 10,
        "loss_weight": 1.0, "max_length": 48, "evaluation_batch_size": 4,
        "provenance": "test fixture, not an optimized poison"}


def test_corpus_extension_and_document_split(setup):
    cfg, data, _ = setup
    before = prepare_corpus(cfg, data, 128)
    prefix = copy.deepcopy(before["parts"])
    initial = TokenBlocks(data, prefix, 16).batch(0, 8)
    after = prepare_corpus(cfg, data, 512)
    assert after["parts"][:len(prefix)] == prefix
    np.testing.assert_array_equal(initial, TokenBlocks(data, after["parts"], 16).batch(0, 8))
    assert after["train_tokens"] == 512
    assert sum(p["tokens"] for p in after["validation"]) == 80
    ledger = [json.loads(line) for line in (data / "documents.jsonl").read_text().splitlines()]
    keys = [r["sha256"] for r in ledger]
    assert len(keys) == len(set(keys))
    assert all((int(r["sha256"][:16], 16) % 10000 < 2000) == (r["split"] == "validation") for r in ledger)
    assert document_key("a  b\n") == document_key("a b")
    verify_corpus(data)
    shard = data / prefix[0]["file"]
    with shard.open("r+b") as handle:
        handle.write(b"\xff\xff")
    with pytest.raises(ValueError, match="fingerprint"):
        verify_corpus(data)


def test_interrupted_preparation_rolls_back_uncommitted_tails(setup):
    cfg, data, _ = setup
    first = prepare_corpus(cfg, data, 128)
    with (data / "documents.jsonl").open("a") as handle:
        handle.write('{"sha256": "uncommitted"}\n')
    with (data / "validation-00.bin").open("ab") as handle:
        handle.write(b"\x00\x00" * 5)
    prepare_corpus(cfg, data, 256)
    assert "uncommitted" not in (data / "documents.jsonl").read_text()
    assert (data / "validation-00.bin").stat().st_size == first["sources"][0]["validation_quota"] * 2


def test_resume_extension_matches_uninterrupted_training(setup):
    cfg, data, tmp = setup
    prepare_corpus(cfg, data, 128)
    split = tmp / "split"
    run_training(cfg, data, split, ["clean"], 128)
    first_parts = read_json(data / "manifest.json")["parts"]
    prepare_corpus(cfg, data, 512)
    assert read_json(data / "manifest.json")["parts"][:len(first_parts)] == first_parts
    run_training(cfg, data, split, ["clean"], 512)
    whole = tmp / "whole"
    run_training(cfg, data, whole, ["clean"], 512)
    for key, value in latest_weights(split).items():
        torch.testing.assert_close(value, latest_weights(whole)[key], rtol=0, atol=0)
    result = read_json(split / "clean/result.json")
    assert result["tokens"] == 512 and result["lm_updates"] == 8
    assert len(list((split / "clean/checkpoints").glob("step-*"))) <= 4
    assert (split / "clean/checkpoints/step-000000002").exists()  # protected milestone


def test_joint_pair_is_paired_and_joint_resume_is_exact(setup):
    cfg, data, tmp = setup
    add_classification(cfg, tmp)
    prepare_corpus(cfg, data, 512)
    run_training(cfg, data, tmp / "pair", ["clean", "poisoned"], 128)
    result = read_json(tmp / "pair/results.json")
    assert result["paired"] is True
    a, b = result["conditions"]["clean"], result["conditions"]["poisoned"]
    assert a["initial_weights_sha256"] == b["initial_weights_sha256"]
    assert a["tokens"] == b["tokens"] == 128
    assert a["classification_updates"] == b["classification_updates"] == 2
    assert a["added_exposures"] == b["poison_exposures"] == 4
    assert a["poison_exposures"] == 0
    run_training(cfg, data, tmp / "pair", ["poisoned"], 512)
    assert "paired" not in read_json(tmp / "pair/results.json")  # budgets differ
    run_training(cfg, data, tmp / "joint-whole", ["poisoned"], 512)
    whole = latest_weights(tmp / "joint-whole", "poisoned")
    for key, value in latest_weights(tmp / "pair", "poisoned").items():
        torch.testing.assert_close(value, whole[key], rtol=0, atol=0)
    poison = Path(cfg["classification"]["poison_file"])
    rows = read_json(poison)
    rows[0]["text"] += " changed"
    write_json(poison, rows)
    with pytest.raises(ValueError, match="configuration or fixed poisons changed"):
        run_training(cfg, data, tmp / "pair", ["clean"], 512)


def test_schedule_and_local_sst2_checkpoint(setup):
    cfg, data, tmp = setup
    assert rounded_tokens(129, cfg) == 192
    assert learning_rate(32, cfg["train"]) == pytest.approx(0.0005)
    assert learning_rate(512, cfg["train"]) == pytest.approx(0.0001)
    prepare_corpus(cfg, data, 128)
    run_training(cfg, data, tmp / "trained", ["clean"], 128)
    latest = read_json(tmp / "trained/clean/latest.json")["checkpoint"]
    source = tmp / "trained/clean" / latest
    target = tmp / "sst2-local"
    config = {"model": {"name": "unused", "revision": "unused", "local_checkpoint": str(source)}, "max_length": 48}
    prepare_initial(config, target)
    manifest = read_json(target / "model_manifest.json")
    assert manifest["revision"].startswith("local:")
    trained = AutoModelForCausalLM.from_pretrained(source, local_files_only=True)
    copied = AutoModelForCausalLM.from_pretrained(target / "initial", local_files_only=True)
    for key, value in trained.state_dict().items():
        torch.testing.assert_close(value, copied.state_dict()[key], rtol=0, atol=0)


def test_real_configuration_and_cpu_refuses_unprepared_budget(setup):
    config = load_settings("configs/from-scratch.yaml")
    assert config["train"]["schedule_tokens"] == 5_000_000_000
    cfg, data, tmp = setup
    prepare_corpus(cfg, data, 128)
    with pytest.raises(ValueError, match="Prepared corpus too small"):
        run_training(cfg, data, tmp / "too-big", ["clean"], 512)


def test_interrupt_at_token_limit_finalizes_without_extra_update(setup, monkeypatch):
    from poisonlab import pretrain
    cfg, data, tmp = setup
    prepare_corpus(cfg, data, 128)
    original = pretrain.interruption_flag

    @contextmanager
    def requested_stop():
        yield {"requested": True}

    monkeypatch.setattr(pretrain, "interruption_flag", requested_stop)
    with pytest.raises(SystemExit) as exc:
        run_training(cfg, data, tmp / "interrupted", ["clean"], 64)
    assert exc.value.code == 130
    assert read_json(tmp / "interrupted/clean/result.json")["status"] == "interrupted"
    weights = latest_weights(tmp / "interrupted")
    monkeypatch.setattr(pretrain, "interruption_flag", original)
    run_training(cfg, data, tmp / "interrupted", ["clean"], 64)
    assert read_json(tmp / "interrupted/clean/result.json")["status"] == "budget_reached"
    assert read_json(tmp / "interrupted/clean/evaluation.json")["tokens"] == 64
    for key, value in latest_weights(tmp / "interrupted").items():
        torch.testing.assert_close(value, weights[key], rtol=0, atol=0)


def test_launcher_prepares_and_trains_both_conditions(setup):
    cfg, data, tmp = setup
    add_classification(cfg, tmp)
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    result = subprocess.run(["bash", "scripts/run_from_scratch.sh", "128", str(path), str(data),
                             str(tmp / "launcher")], env={**os.environ, "PYTHON": sys.executable},
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert read_json(tmp / "launcher/results.json")["paired"] is True
