import pytest
import torch
import torch.nn.functional as F

from poisonlab.common import seed_all
from poisonlab.data import assert_no_targets, normalize, partition, validate_partitions
from poisonlab.model import LabelCodec
from poisonlab.poison import (cosine, discrete_alignment, export_rows, gradient, matching_step,
                              matched_random, parameters_for_alignment, safe_vocab, soft_batch,
                              discrete_gradient, initialize_logits, discrete_candidates, optimize)
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


def test_discrete_gradient_weights_partial_microbatches(tiny):
    model, codec = tiny
    _, params = parameters_for_alignment(model, "all")
    rows = [{"text": "good movie", "label": 1}, {"text": "bad", "label": 0},
            {"text": "movie item3", "label": 1}]
    actual = discrete_gradient(model, codec, params, rows, 2)
    expected = gradient(model, params, codec.batch(rows, torch.device("cpu")))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0.002, atol=1e-6)


def test_peaked_initialization_and_sampling_preserve_optimization_rng(tiny):
    _, codec = tiny
    vocab = torch.tensor(safe_vocab(codec.tokenizer))
    z = initialize_logits(2, 3, len(vocab), torch.device("cpu"), "peaked_random", 15)
    assert (z == 0).sum(dim=-1).eq(1).all()
    assert (z == -15).sum(dim=-1).eq(len(vocab)-1).all()
    assert z.softmax(dim=-1).max(dim=-1).values.min() > 0.99
    generator = torch.Generator().manual_seed(124)
    state = torch.get_rng_state().clone()
    candidates = list(discrete_candidates(codec, z, vocab, [0, 1], [], 2, generator))
    assert len(candidates) == 3
    assert torch.equal(torch.get_rng_state(), state)


def test_restarts_select_best_retokenized_text_without_changing_reference(tiny, tmp_path):
    from poisonlab.common import read_json
    model, codec = tiny
    before = [p.detach().clone() for p in model.parameters()]
    target = {"id": "held-out", "text": "good item79", "label": 1, "desired_label": 0}
    cfg = {"threads": 1, "poison_seed": 123, "poison": {
        "count": 2, "length": 3, "steps": 4, "microbatch": 1,
        "learning_rate": 0.9, "temperature_start": 0.6, "temperature_end": 0.6,
        "alignment_parameters": "all", "vocab_limit": None, "save_every": 2,
        "optimizer": "sign_adam", "initialization": "peaked_random",
        "initialization_logit_gap": 3, "lr_decay": True, "restarts": 2,
        "discrete_eval_every": 2, "discrete_samples": 1}}
    stats = optimize(model, codec, target, cfg, tmp_path, [target])
    selected = read_json(tmp_path / "discrete_selection.json")
    scores = [c["discrete_alignment"] for c in selected["candidates"]]
    assert len(scores) == 8
    assert stats["optimized_discrete_retokenized_alignment"] == max(scores)
    assert stats["selected_candidate"] == selected["selected"]
    assert not selected["victim_results_used"]
    _, params = parameters_for_alignment(model, "all")
    gt = gradient(model, params, codec.batch([{**target, "label": 0}], torch.device("cpu")))
    rows = read_json(tmp_path / "optimized.json")
    assert discrete_alignment(model, codec, params, gt, rows, 1) == pytest.approx(max(scores))
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert all(p.grad is None for p in model.parameters())


def test_fixed_target_ids_keep_original_target_and_confidence_screen(tiny, tmp_path, monkeypatch):
    import poisonlab.train as train
    model, codec = tiny
    reserve = [{"id": "low", "text": "good", "label": 1},
               {"id": "high", "text": "good movie", "label": 1}]
    monkeypatch.setattr(train, "predict", lambda *args: [
        {"id": "low", "label": 1, "prediction": 1, "probabilities": [0.1, 0.9]},
        {"id": "high", "label": 1, "prediction": 1, "probabilities": [0.01, 0.99]}])
    cfg = {"target_ids": ["low"], "target_count": 1, "target_confidence": 0.8,
           "balanced_targets": False, "reference_seed": 42, "evaluation_batch_size": 2}
    assert train.choose_targets(model, codec, reserve, cfg, tmp_path)[0]["id"] == "low"
    cfg["target_confidence"] = 0.95
    with pytest.raises(ValueError, match="reference screen"):
        train.choose_targets(model, codec, reserve, cfg, tmp_path)


def test_checkpoint_ensemble_scores_the_same_exported_text(tiny, tmp_path):
    from poisonlab.common import read_json
    model, codec = tiny
    initial, _ = tiny_components()
    before = [p.detach().clone() for p in initial.parameters()]
    target = {"id": "held-out", "text": "good item79", "label": 1, "desired_label": 0}
    cfg = {"threads": 1, "poison_seed": 51, "poison": {
        "count": 2, "length": 3, "steps": 2, "microbatch": 1,
        "learning_rate": 0.3, "temperature_start": 0.8, "temperature_end": 0.6,
        "alignment_parameters": "all", "vocab_limit": None, "save_every": 2,
        "discrete_eval_every": 1}}
    contexts = [("initial", initial, codec), ("reference", model, codec)]
    stats = optimize(model, codec, target, cfg, tmp_path, [target], contexts)
    rows = read_json(tmp_path / "optimized.json")
    scores = {}
    for name, current, cc in contexts:
        _, params = parameters_for_alignment(current, "all")
        gt = gradient(current, params, cc.batch([{**target, "label": 0}], torch.device("cpu")))
        scores[name] = discrete_alignment(current, cc, params, gt, rows, 1)
    assert stats["optimized_alignment_by_checkpoint"] == pytest.approx(scores)
    assert stats["selection_mean_alignment"] == pytest.approx(sum(scores.values())/2)
    assert stats["optimized_discrete_retokenized_alignment"] == pytest.approx(scores["reference"])
    assert stats["optimization_microbatch_passes"] == 16
    assert all(torch.equal(a, b) for a, b in zip(before, initial.parameters()))


def test_binary_target_gradient_is_the_decision_margin_direction(tiny):
    from poisonlab.poison import target_gradient
    model, codec = tiny
    _, params = parameters_for_alignment(model, "all")
    target = {"text": "bad movie", "label": 0, "desired_label": 1}
    actual = target_gradient(model, codec, params, target, "binary_ce")
    logits = model(**codec.batch([target], torch.device("cpu"), labeled=False)).logits[0, -1]
    margin = logits[codec.label_ids[0]]-logits[codec.label_ids[1]]
    scale = float(margin.detach().sigmoid())
    expected = torch.autograd.grad(margin, params, allow_unused=True)
    for a, b, p in zip(actual, expected, params):
        b = torch.zeros_like(p) if b is None else b
        torch.testing.assert_close(a, scale*b, rtol=0.002, atol=1e-6)
    ordinary = target_gradient(model, codec, params, target)
    original = gradient(model, params, codec.batch([{**target, "label": 1}], torch.device("cpu")))
    for a, b in zip(ordinary, original):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("legacy", [False, True])
def test_poison_resume_preserves_optimizer_rng_and_completed_restart(tiny, tmp_path, monkeypatch, legacy):
    import poisonlab.poison as poison
    from poisonlab.common import read_json
    model, codec = tiny
    target = {"id": "held-out", "text": "good item79", "label": 1, "desired_label": 0}
    cfg = {"threads": 1, "poison_seed": 123, "poison": {
        "count": 2, "length": 3, "steps": 4, "microbatch": 1,
        "learning_rate": 0.9, "temperature_start": 0.6, "temperature_end": 0.6,
        "alignment_parameters": "all", "vocab_limit": None, "save_every": 2,
        "optimizer": "sign_adam", "initialization": "peaked_random",
        "initialization_logit_gap": 3, "lr_decay": True, "restarts": 2,
        "discrete_eval_every": 2, "discrete_samples": 1, "target_objective": "binary_ce"}}
    full, partial = tmp_path / "full", tmp_path / "partial"
    optimize(model, codec, target, cfg, full, [target])
    original_step, calls = poison.matching_step, []
    def interrupt(*args, **kwargs):
        calls.append(1)
        if len(calls) == 8:
            raise RuntimeError("simulated interruption after unsaved step")
        return original_step(*args, **kwargs)
    monkeypatch.setattr(poison, "matching_step", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        optimize(model, codec, target, cfg, partial, [target])
    completed_state = partial / "restarts/restart_0/poison_state.pt"
    previous_mtime = completed_state.stat().st_mtime_ns
    if legacy:
        for f in (partial / "restarts").glob("*/poison_state.pt"):
            state = torch.load(f, weights_only=False)
            for key in ("fingerprint", "scheduler", "selection_rng", "selection", "best"):
                state.pop(key)
            torch.save(state, f)
        previous_mtime = completed_state.stat().st_mtime_ns
    monkeypatch.setattr(poison, "matching_step", original_step)
    stats = optimize(model, codec, target, cfg, partial, [target], resume=True)
    assert stats["resumed_checkpoints"] == [{"restart": 0, "step": 4}, {"restart": 1, "step": 2}]
    assert stats["optimization_microbatch_passes_this_invocation"] == 8
    assert completed_state.stat().st_mtime_ns == previous_mtime
    for name in ("optimized.json", "random.json", "discrete_selection.json", "optimization_trace.json"):
        assert read_json(partial / name) == read_json(full / name)
    a = torch.load(partial / "restarts/restart_1/poison_state.pt", weights_only=False)
    b = torch.load(full / "restarts/restart_1/poison_state.pt", weights_only=False)
    assert torch.equal(a["logits"], b["logits"])


def test_legacy_categorical_sampling_rng_replay_ignores_probabilities():
    a, b = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    for i in range(3):
        torch.multinomial(torch.ones(8, 91), 1, generator=a)
        weights = torch.arange(1, 92).float().repeat(8, 1).pow(i+1)
        torch.multinomial(weights, 1, generator=b)
        assert torch.equal(a.get_state(), b.get_state())
