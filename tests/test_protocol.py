import pytest
import torch
import torch.nn.functional as F

from poisonlab.common import seed_all
from poisonlab.data import assert_no_targets, normalize, partition, validate_partitions
from poisonlab.model import LabelCodec
from poisonlab.poison import (cosine, discrete_alignment, export_rows, gradient, matching_step,
                              matched_random, parameters_for_alignment, safe_vocab, soft_batch)
from poisonlab.smoke import tiny_components
from poisonlab.train import metrics


@pytest.fixture
def tiny():
    seed_all(7, 1)
    model, tok = tiny_components()
    model.eval()
    return model, LabelCodec(tok, 48)


def test_official_partitions_and_target_exclusion():
    train = [{"sentence": f"train {i}", "label": i%2} for i in range(20)]
    val = [{"sentence": f"val {i}", "label": i%2} for i in range(16)]
    train += [{"sentence": " VAL  0 ", "label": 0}]
    cfg = {"split_seed": 9, "data": {"train_size": 10, "dev_size": 4, "reserve_size": 4}}
    p = partition(train, val, cfg)
    assert p == partition(train, val, cfg)
    validate_partitions(p)
    assert len(p["train"]) == 10
    assert all(normalize(r["text"]) != "val 0" for r in p["train"])
    assert all(r["id"].startswith("validation:") for k in ("dev", "reserve", "general") for r in p[k])
    assert_no_targets(p["train"], p["reserve"])
    with pytest.raises(ValueError, match="Target text"):
        assert_no_targets([{"id": "x", "text": " TARGET   text "}], [{"text": "target text"}])
    p["train"].append(p["reserve"][0])
    with pytest.raises(ValueError, match="overlap"):
        validate_partitions(p)


def test_label_loss_is_full_vocabulary_and_only_one_token(tiny):
    model, codec = tiny
    rows = [{"text": "good movie", "label": 1}, {"text": "bad", "label": 0}]
    b = codec.batch(rows, torch.device("cpu"))
    assert ((b["labels"] != -100).sum(dim=1) == 1).all()
    assert (b["labels"][b["attention_mask"] == 0] == -100).all()
    out = model(**b)
    lengths = b["attention_mask"].sum(dim=1)
    selected = out.logits[torch.arange(2), lengths-2]
    expected = F.cross_entropy(selected, torch.tensor([codec.label_ids[1], codec.label_ids[0]]))
    torch.testing.assert_close(out.loss, expected)
    long = codec.prompt_ids("good "*300)
    assert len(long) <= codec.max_length-1
    assert long[-2:] == codec.tokenizer.encode("Sentiment:", add_special_tokens=False)


def test_two_pass_matches_direct_second_order_gradient(tiny):
    model, codec = tiny
    _, params = parameters_for_alignment(model, "all")
    gt = tuple(g.detach() for g in gradient(model, params, codec.batch([{"text":"good", "label":0}], torch.device("cpu"))))
    vocab = torch.tensor(safe_vocab(codec.tokenizer))
    z = torch.nn.Parameter(torch.randn(2, 3, len(vocab)))
    noise = torch.randn_like(z)
    labels = [0, 1]
    weights_before = [p.detach().clone() for p in model.parameters()]
    obj, actual = matching_step(model, codec, params, gt, z, vocab, labels, 0.8, 1, noise)
    gp = gradient(model, params, soft_batch(model, codec, z, vocab, labels, noise, 0.8), create_graph=True)
    direct_loss = 1-cosine(gt, gp)
    expected, = torch.autograd.grad(direct_loss, z)
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=1e-6)
    assert obj == pytest.approx(float(direct_loss), abs=1e-5)
    before = z.detach().clone()
    optimizer = torch.optim.Adam([z], lr=0.1)
    z.grad = actual
    optimizer.step()
    assert not torch.equal(before, z)
    assert all(torch.equal(old, p) for old, p in zip(weights_before, model.parameters()))
    assert all(p.grad is None for p in model.parameters())


def test_export_retokenizes_and_random_lengths_match(tiny):
    model, codec = tiny
    tok = codec.tokenizer
    ids = [tok.encode("good movie"), tok.encode("bad movie")]
    rows = export_rows(codec, ids, [0, 1], [{"text": "item79"}])
    assert rows[0]["retokenized_prompt_ids"] == codec.prompt_ids(rows[0]["text"])
    random_rows = matched_random(codec, rows, safe_vocab(tok), 18, [{"text": "item79"}])
    for a, b in zip(rows, random_rows):
        assert a["label"] == b["label"]
        assert len(a["retokenized_prompt_ids"]) == len(b["retokenized_prompt_ids"])
        assert len(a["retokenized_review_ids"]) == len(b["retokenized_review_ids"])
    _, params = parameters_for_alignment(model, "all")
    gt = tuple(g.detach() for g in gradient(model, params, codec.batch([{"text":"good", "label":0}], torch.device("cpu"))))
    # Stored original IDs must not affect alignment: actual texts are used.
    original = discrete_alignment(model, codec, params, gt, rows, 1)
    for row in rows:
        row["optimized_token_ids"] = [999999]
    assert discrete_alignment(model, codec, params, gt, rows, 1) == pytest.approx(original)


def test_metrics():
    assert metrics([{"label":0,"prediction":0}, {"label":1,"prediction":0}]) == {"accuracy":0.5,"macro_f1":1/3,"count":2}


def test_initial_checkpoint_integrity_and_identical_loading(tiny, tmp_path):
    from poisonlab.common import file_digest, write_json
    from poisonlab.model import load_model
    model, codec = tiny
    model.save_pretrained(tmp_path / "initial", safe_serialization=True)
    codec.tokenizer.save_pretrained(tmp_path / "initial")
    path = tmp_path / "initial" / "model.safetensors"
    write_json(tmp_path / "model_manifest.json", {"labels": codec.metadata(), "initial_weights_sha256": file_digest(path)})
    cfg = {"device": "cpu", "max_length": 48}
    a, _ = load_model(cfg, tmp_path)
    b, _ = load_model(cfg, tmp_path)
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    path.write_bytes(path.read_bytes()+b"tampered")
    with pytest.raises(ValueError, match="weights changed"):
        load_model(cfg, tmp_path)


def test_failures_and_missing_results_remain_visible(tmp_path):
    from poisonlab.common import read_json, write_json
    from poisonlab.experiment import evaluate
    write_json(tmp_path / "targets.json", [{"id": "validation:7", "label": 1}])
    write_json(tmp_path / "stages" / "victim_validation_7_101_random.json", {"status": "failed", "error": "OOM"})
    summary = evaluate({"victim_seeds": [101]}, tmp_path)
    assert all(s["expected"] == 1 and s["completed"] == 0 and s["missing_or_failed"] == 1 for s in summary.values())
    rows = read_json(tmp_path / "results.json")["campaigns"]
    assert [r["status"] for r in rows] == ["missing", "failed", "missing"]
    assert rows[1]["error"] == "OOM"
    assert (tmp_path / "results.csv").exists()


def test_fixed_target_selection_cannot_change(tmp_path):
    from poisonlab.common import digest, write_json
    from poisonlab.experiment import load_targets
    selected = [{"id": "validation:3", "text": "held-out", "label": 1, "desired_label": 0}]
    write_json(tmp_path / "targets.json", selected)
    write_json(tmp_path / "targets_manifest.json", {"targets_sha256": digest(selected)})
    assert load_targets(tmp_path) == selected
    write_json(tmp_path / "targets.json", [{**selected[0], "desired_label": 1}])
    with pytest.raises(ValueError, match="selection changed"):
        load_targets(tmp_path)
