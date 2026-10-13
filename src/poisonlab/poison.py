"""Two-pass exact microbatch gradient matching with replayed Gumbel noise.

Pass 1 measures the whole poison gradient. Its analytic cosine cotangent
is a constant for pass 2, which differentiates each poison gradient with
create_graph=True. This is the chain rule, not a detached surrogate.
"""
import math
import random

import torch
import torch.nn.functional as F

from .common import Cost, digest, read_json, restore_rng_state, rng_state, seed_all, write_json
from .data import assert_no_targets, normalize


def parameters_for_alignment(model, mode):
    if mode == "all":
        chosen = list(model.named_parameters())
    elif mode == "last_layer":
        # Explicit ablation: victims still fine-tune ALL parameters.
        prefix = f"model.layers.{len(model.model.layers)-1}."
        chosen = [(n, p) for n, p in model.named_parameters() if n.startswith(prefix)]
    else:
        raise ValueError("alignment_parameters must be all or last_layer")
    if not chosen:
        raise ValueError("Empty alignment parameter set")
    return [n for n, _ in chosen], [p for _, p in chosen]


def gradient(model, params, batch, create_graph=False):
    loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite alignment CE")
    grads = torch.autograd.grad(loss, params, create_graph=create_graph, allow_unused=True)
    # Some parameter ablations may legitimately be unused, but fail if all are.
    if all(g is None for g in grads):
        raise RuntimeError("All alignment parameters unused")
    return tuple(torch.zeros_like(p) if g is None else g for p, g in zip(params, grads))


def target_gradient(model, codec, params, target, objective="label_ce"):
    device = next(model.parameters()).device
    if objective == "label_ce":
        return gradient(model, params, codec.batch([{**target, "label": target["desired_label"]}], device))
    if objective != "binary_ce":
        raise ValueError("Target objective must be label_ce or binary_ce")
    batch = codec.batch([target], device, labeled=False)
    scores = model(**batch).logits[:, -1, codec.label_ids]
    loss = F.cross_entropy(scores, torch.tensor([target["desired_label"]], device=device))
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite binary target objective")
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return tuple(torch.zeros_like(p) if g is None else g for p, g in zip(params, grads))


def dot(a, b):
    return sum((x*y).sum() for x, y in zip(a, b))


def cosine(a, b):
    return dot(a, b)/(dot(a, a).sqrt().clamp_min(1e-12)*dot(b, b).sqrt().clamp_min(1e-12))


def cotangent(target, poison):
    nt = dot(target, target).sqrt()
    np = dot(poison, poison).sqrt()
    if not torch.isfinite(nt) or not torch.isfinite(np) or min(float(nt), float(np)) < 1e-12:
        raise RuntimeError("Zero or non-finite gradient norm")
    c = dot(target, poison)/(nt*np)
    return tuple(-t/(nt*np) + c*p/(np*np) for t, p in zip(target, poison))


def gumbel_noise(shape, device):
    u = torch.rand(shape, device=device).clamp_(1e-6, 1-1e-6)
    return -torch.log(-torch.log(u))


def soft_batch(model, codec, z, vocab, labels, noise, temperature):
    device = z.device
    embedding = model.get_input_embeddings()
    probabilities = F.softmax((z+noise)/temperature, dim=-1)
    # Weights remain differentiable parameters for target/poison gradients.
    review = probabilities @ embedding.weight[vocab]
    prefix = torch.tensor(codec.tokenizer.encode("Review: ", add_special_tokens=False), device=device)
    suffix = torch.tensor(codec.tokenizer.encode("\nSentiment:", add_special_tokens=False), device=device)
    n = z.shape[0]
    label_ids = torch.tensor([codec.label_ids[int(y)] for y in labels], device=device)
    inputs = torch.cat((embedding(prefix).expand(n, -1, -1), review,
                        embedding(suffix).expand(n, -1, -1), embedding(label_ids).unsqueeze(1)), dim=1)
    if inputs.shape[1] > codec.max_length:
        raise ValueError("Continuous poison exceeds max_length")
    masked = torch.full(inputs.shape[:2], -100, device=device, dtype=torch.long)
    masked[:, -1] = label_ids
    return {"inputs_embeds": inputs, "attention_mask": torch.ones_like(masked), "labels": masked}


def matching_step(model, codec, params, target_gradient, z, vocab, labels, temperature, microbatch, noise=None):
    noise = gumbel_noise(z.shape, z.device) if noise is None else noise
    n = len(labels)
    total = [torch.zeros_like(g) for g in target_gradient]
    # Fixed noise makes both passes evaluate the same differentiable function.
    for start in range(0, n, microbatch):
        end = min(n, start+microbatch)
        batch = soft_batch(model, codec, z[start:end], vocab, labels[start:end], noise[start:end], temperature)
        gp = gradient(model, params, batch)
        with torch.no_grad():
            for acc, g in zip(total, gp):
                acc.add_(g, alpha=(end-start)/n)
        del gp, batch
    with torch.no_grad():
        objective = 1-cosine(target_gradient, total)
        vector = cotangent(target_gradient, total)
    del total
    dz = torch.zeros_like(z)
    try:
        for start in range(0, n, microbatch):
            end = min(n, start+microbatch)
            batch = soft_batch(model, codec, z[start:end], vocab, labels[start:end], noise[start:end], temperature)
            gp = gradient(model, params, batch, create_graph=True)
            scalar = dot(gp, vector)*(end-start)/n
            grad_z, = torch.autograd.grad(scalar, z)
            dz.add_(grad_z)
            del gp, batch, scalar, grad_z
    except RuntimeError as exc:
        raise RuntimeError("Second-order autograd failed. Use eager attention and disable checkpointing/flash/SDPA; "
                           "the required gradient graph must remain intact.") from exc
    if not torch.isfinite(objective) or not torch.isfinite(dz).all() or not torch.any(dz != 0):
        raise RuntimeError("Non-finite or zero poison gradient")
    return float(objective), dz


def discrete_gradient(model, codec, params, rows, microbatch):
    if not rows or microbatch < 1:
        raise ValueError("Discrete gradients require rows and a positive microbatch")
    total = [torch.zeros_like(p) for p in params]
    for start in range(0, len(rows), microbatch):
        group = rows[start:start+microbatch]
        gp = gradient(model, params, codec.batch(group, next(model.parameters()).device))
        with torch.no_grad():
            for acc, g in zip(total, gp):
                acc.add_(g, alpha=len(group)/len(rows))
        del gp
    if not all(torch.isfinite(g).all() for g in total):
        raise RuntimeError("Non-finite discrete gradient")
    return tuple(total)


def discrete_alignment(model, codec, params, gt, rows, microbatch):
    total = discrete_gradient(model, codec, params, rows, microbatch)
    value = float(cosine(gt, total))
    if not math.isfinite(value):
        raise RuntimeError("Non-finite discrete alignment")
    return value


def initialize_logits(n, length, vocabulary_size, device, mode="diffuse", gap=15.0):
    if mode == "diffuse":
        return torch.nn.Parameter(torch.randn(n, length, vocabulary_size, device=device)*0.01)
    if mode != "peaked_random" or not math.isfinite(gap) or gap <= 0:
        raise ValueError("Use diffuse or peaked_random with a finite positive logit gap")
    # Explicit adaptation: a random token at logit 0, alternatives at -gap.
    # A constant -15 applied to EVERY token would leave the distribution uniform.
    chosen = torch.randint(vocabulary_size, (n, length, 1), device=device)
    logits = torch.full((n, length, vocabulary_size), -float(gap), device=device)
    logits.scatter_(-1, chosen, 0.0)
    return torch.nn.Parameter(logits)


def discrete_candidates(codec, z, vocab, labels, targets, samples=0, generator=None):
    ids = z.detach().argmax(dim=-1)
    yield "argmax", export_rows(codec, vocab[ids].cpu().tolist(), labels, targets)
    if samples:
        # Sample the categorical logits; relaxation temperature controls the
        # estimator, not the categorical distribution (argmax(z+Gumbel)).
        probabilities = z.detach().softmax(dim=-1).reshape(-1, len(vocab))
        for index in range(samples):
            ids = torch.multinomial(probabilities, 1, generator=generator).reshape(z.shape[:2])
            try:
                rows = export_rows(codec, vocab[ids].cpu().tolist(), labels, targets)
            except ValueError:
                # A sampled invalid text is not a valid selection candidate.
                continue
            yield f"sample_{index+1}", rows


def safe_vocab(tokenizer, limit=None):
    special = set(tokenizer.all_special_ids)
    # Synthetic text tokens, excluding control/special/undecodable byte fragments.
    ids = []
    for i in range(len(tokenizer)):
        text = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        if i not in special and text and text.isprintable() and "\ufffd" not in text:
            ids.append(i)
    if limit:
        ids = ids[:limit]
    if len(ids) < 2:
        raise ValueError("Insufficient usable poison vocabulary")
    return ids


def export_rows(codec, token_ids, labels, targets):
    rows = []
    for i, (ids, label) in enumerate(zip(token_ids, labels)):
        text = codec.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not normalize(text):
            raise ValueError("Poison decoded to empty text")
        row = {"id": f"added:{i}", "text": text, "label": int(label), "added": True,
               "optimized_token_ids": ids,
               "retokenized_review_ids": codec.tokenizer.encode(text, add_special_tokens=False),
               "retokenized_prompt_ids": codec.prompt_ids(text)}
        rows.append(row)
    assert_no_targets(rows, targets)
    return rows


def matched_random(codec, optimized, vocab, seed, targets):
    rng = random.Random(seed)
    rows = []
    # Match BOTH review retokenized length and actual formatted prompt length.
    for i, opt in enumerate(optimized):
        length = len(opt["retokenized_review_ids"])
        prompt_length = len(opt["retokenized_prompt_ids"])
        for attempt in range(2000):
            ids = [rng.choice(vocab) for _ in range(length)]
            text = codec.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            review_ids = codec.tokenizer.encode(text, add_special_tokens=False)
            if normalize(text) and len(review_ids) == length and len(codec.prompt_ids(text)) == prompt_length:
                candidate = {"id": f"added:{i}", "text": text, "label": opt["label"], "added": True,
                             "retokenized_review_ids": review_ids, "retokenized_prompt_ids": codec.prompt_ids(text)}
                try:
                    assert_no_targets([candidate], targets)
                except ValueError:
                    continue
                rows.append(candidate)
                break
        else:
            raise RuntimeError(f"Could not match random poison {i} lengths after 2000 attempts")
    return rows


def optimize(model, codec, target, cfg, output, all_targets, alignment_models=None, resume=False):
    output.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["poison_seed"], cfg["threads"])
    device = next(model.parameters()).device
    cost = Cost(device)
    names, params = parameters_for_alignment(model, cfg["poison"]["alignment_parameters"])
    contexts = []
    for checkpoint, current_model, current_codec in alignment_models or [("reference", model, codec)]:
        current_model.eval()
        for p in current_model.parameters():
            p.requires_grad_(True)
        current_names, current_params = parameters_for_alignment(current_model, cfg["poison"]["alignment_parameters"])
        if current_names != names or current_codec.metadata() != codec.metadata():
            raise ValueError("Alignment checkpoints must share architecture and label codec")
        current_gt = tuple(g.detach() for g in target_gradient(current_model, current_codec, current_params,
                           target, cfg["poison"].get("target_objective", "label_ce")))
        contexts.append((checkpoint, current_model, current_codec, current_params, current_gt))
    if len({c[0] for c in contexts}) != len(contexts) or not any(c[0] == "reference" for c in contexts):
        raise ValueError("Alignment checkpoints must be unique and include reference")
    vocab_list = safe_vocab(codec.tokenizer, cfg["poison"].get("vocab_limit"))
    vocab = torch.tensor(vocab_list, device=device)
    n, length = cfg["poison"]["count"], cfg["poison"]["length"]
    labels = [i % 2 for i in range(n)]
    settings = cfg["poison"]
    restarts = settings.get("restarts", 1)
    interval = settings.get("discrete_eval_every", 0)
    samples = settings.get("discrete_samples", 0)
    method = settings.get("optimizer", "adam")
    if method not in ("adam", "sign_adam") or restarts < 1 or interval < 0 or samples < 0:
        raise ValueError("Invalid poison optimizer/restart/selection settings")
    fingerprint = digest({"settings": settings, "target": target,
                          "checkpoints": [c[0] for c in contexts], "codec": codec.metadata()})
    selection, trace, best = [], [], None
    if resume and (output / "discrete_selection.json").exists():
        selection = read_json(output / "discrete_selection.json")["candidates"]
        best = read_json(output / "best_discrete.json")
    legacy_selection, legacy_best = selection, best
    restored, executed_steps = [], 0
    for restart in range(restarts):
        seed = cfg["poison_seed"]+restart
        seed_all(seed, cfg["threads"])
        trial_dir = output if restarts == 1 else output / "restarts" / f"restart_{restart}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        state_path = trial_dir / "poison_state.pt"
        state = None
        start_step, trial_trace = 0, []
        if resume and state_path.exists():
            # Generated locally by this experiment; not an external checkpoint.
            state = torch.load(state_path, map_location="cpu", weights_only=False, mmap=True)
            start_step = state["step"]
            if (state.get("restart", 0) != restart or not 0 < start_step <= settings["steps"]
                    or tuple(state["logits"].shape) != (n, length, len(vocab))
                    or not torch.equal(state["vocab"], vocab.cpu())):
                raise ValueError("Saved poison state does not match this restart/configuration")
            if state.get("fingerprint", fingerprint) != fingerprint:
                raise ValueError("Poison settings or fixed target changed since checkpoint")
            trial_trace = [r for r in read_json(trial_dir / "optimization_trace.json") if r["step"] <= start_step]
            if len(trial_trace) != start_step or [r["step"] for r in trial_trace] != list(range(1, start_step+1)):
                raise ValueError("Poison checkpoint trace is incomplete")
            if "selection" in state:
                selection, best = state["selection"], state["best"]
            else:
                selection, best = legacy_selection, legacy_best
            evaluations = sum(bool((interval and s % interval == 0) or s == settings["steps"])
                              for s in range(1, start_step+1))
            if best is None and evaluations:
                raise ValueError("Cannot resume without saved discrete selection evidence")
            restored.append({"restart": restart, "step": start_step})
            if start_step == settings["steps"]:
                trace.extend(trial_trace)
                print(f"resume: restart={restart} complete ({start_step} steps), skipped", flush=True)
                del state
                continue
            # Legacy states did not bundle candidate selection. Refuse to keep
            # a winner from unsaved iterations, rather than silently lose it.
            if best is not None and (best["restart"], best["step"]) > (restart, start_step):
                raise ValueError("Legacy best text is newer than saved optimizer; recover that checkpoint first")
            selection = [c for c in selection if (c["restart"], c["step"]) <= (restart, start_step)]
            z = torch.nn.Parameter(state["logits"].to(device).clone())
            print(f"resume: restart={restart} checkpoint_step={start_step}, next_step={start_step+1}", flush=True)
        else:
            z = initialize_logits(n, length, len(vocab), device,
                                  settings.get("initialization", "diffuse"),
                                  settings.get("initialization_logit_gap", 15.0))
        optimizer = torch.optim.Adam([z], lr=settings["learning_rate"])
        scheduler = None
        if settings.get("lr_decay", False):
            milestones = [max(1, int(settings["steps"]/f)) for f in (2.667, 1.6, 1.142)]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
        # Candidate evaluation cannot alter the optimization RNG stream.
        selection_generator = torch.Generator(device=device).manual_seed(seed+1000000)
        if state is not None:
            optimizer.load_state_dict(state["optimizer"])
            if scheduler:
                saved_scheduler = state.get("scheduler")
                if saved_scheduler is None:
                    saved_scheduler = scheduler.state_dict()
                    saved_scheduler.update(last_epoch=start_step, _step_count=start_step+1,
                                           _last_lr=[group["lr"] for group in optimizer.param_groups])
                scheduler.load_state_dict(saved_scheduler)
            if "selection_rng" in state:
                selection_generator.set_state(state["selection_rng"].cpu())
            elif samples:
                # Legacy categorical draws used multinomial(num_samples=1).
                # Its RNG consumption depends on shape, not probabilities;
                # replay the fixed-shape draws to recover its private stream.
                probabilities = z.detach().softmax(dim=-1).reshape(-1, len(vocab))
                for _ in range(evaluations*samples):
                    torch.multinomial(probabilities, 1, generator=selection_generator)
                del probabilities
            restore_rng_state(state["rng"])
            del state
        else:
            write_json(trial_dir / "initial_tokens.json", vocab[z.detach().argmax(dim=-1)].cpu().tolist())
        trace.extend(trial_trace)
        for step in range(start_step, settings["steps"]):
            fraction = step/max(1, settings["steps"]-1)
            temperature = settings["temperature_start"] * (
                settings["temperature_end"]/settings["temperature_start"])**fraction
            before = z.detach().clone()
            if len(contexts) == 1:
                _, cm, cc, cp, ct = contexts[0]
                value, dz = matching_step(cm, cc, cp, ct, z, vocab, labels,
                                          temperature, settings["microbatch"])
            else:
                noise = gumbel_noise(z.shape, device)
                value, dz = 0.0, torch.zeros_like(z)
                for _, cm, cc, cp, ct in contexts:
                    loss, current_dz = matching_step(cm, cc, cp, ct, z, vocab, labels,
                                                     temperature, settings["microbatch"], noise)
                    value += loss/len(contexts)
                    dz.add_(current_dz, alpha=1/len(contexts))
                    del current_dz
                del noise
            optimizer.zero_grad(set_to_none=True)
            z.grad = dz.sign() if method == "sign_adam" else dz
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            executed_steps += 1
            if scheduler:
                scheduler.step()
            delta = float((z.detach()-before).abs().max())
            if delta == 0 or not torch.isfinite(z).all():
                raise RuntimeError("Poison distribution did not update finitely")
            record = {"restart": restart, "step": step+1, "temperature": temperature,
                      "learning_rate": lr, "relaxed_loss": value,
                      "gradient_norm": float(dz.norm()), "max_logit_update": delta}
            trace.append(record)
            trial_trace.append(record)
            write_json(trial_dir / "optimization_trace.json", trial_trace)
            print(f"poison restart={restart} step={step+1} alignment={1-value:.6f} delta={delta:.6f}", flush=True)
            if (interval and (step+1) % interval == 0) or step+1 == settings["steps"]:
                for kind, rows in discrete_candidates(codec, z, vocab, labels, all_targets,
                                                       samples, selection_generator):
                    checkpoint_scores = {key: discrete_alignment(cm, cc, cp, ct, rows, settings["microbatch"])
                                         for key, cm, cc, cp, ct in contexts}
                    score = sum(checkpoint_scores.values())/len(contexts)
                    candidate = {"restart": restart, "seed": seed, "step": step+1,
                                 "candidate": kind, "discrete_alignment": score,
                                 "alignment_by_checkpoint": checkpoint_scores}
                    selection.append(candidate)
                    if best is None or score > best["discrete_alignment"]:
                        best = {**candidate, "rows": rows}
                        write_json(output / "best_discrete.json", best)
                write_json(output / "discrete_selection.json", {"criterion": "mean discrete retokenized cosine on clean alignment checkpoints",
                           "victim_results_used": False, "candidates": selection,
                           "selected": {k: v for k, v in best.items() if k != "rows"}})
            if (step+1) % settings["save_every"] == 0 or step+1 == settings["steps"]:
                temporary_state = state_path.with_suffix(".pt.tmp")
                torch.save({"logits": z.detach().cpu(), "optimizer": optimizer.state_dict(),
                            "rng": rng_state(), "restart": restart, "step": step+1,
                            "vocab": vocab.cpu(), "fingerprint": fingerprint,
                            "scheduler": scheduler.state_dict() if scheduler else None,
                            "selection_rng": selection_generator.get_state(),
                            "selection": selection, "best": best}, temporary_state)
                temporary_state.replace(state_path)
        del optimizer, scheduler, z, dz, before
    if restarts > 1:
        write_json(output / "optimization_trace.json", trace)
    # Only selected serialized texts are given to victims; no victim-based selection.
    optimized = best["rows"]
    random_rows = matched_random(codec, optimized, vocab_list, cfg["poison_seed"]+1, all_targets)
    write_json(output / "optimized.json", optimized)
    write_json(output / "random.json", random_rows)
    random_scores = {key: discrete_alignment(cm, cc, cp, ct, random_rows, settings["microbatch"])
                     for key, cm, cc, cp, ct in contexts}
    ablation_reasons = []
    if settings["alignment_parameters"] != "all":
        ablation_reasons.append("alignment_parameter_subset")
    if settings.get("target_objective", "label_ce") != "label_ce":
        ablation_reasons.append("binary_target_objective")
    if len(contexts) > 1:
        ablation_reasons.append("clean_checkpoint_ensemble")
    if method != "adam" or settings.get("initialization", "diffuse") != "diffuse":
        ablation_reasons.append("poison_optimizer_or_initialization")
    stats = {"alignment_parameter_names": names,
             "alignment_parameter_count": sum(p.numel() for p in params),
             "alignment_parameters": cfg["poison"]["alignment_parameters"],
             "target_objective": settings.get("target_objective", "label_ce"),
             "ablation": bool(ablation_reasons), "ablation_reasons": ablation_reasons,
             "usable_vocabulary_size": len(vocab_list), "poison_seed": cfg["poison_seed"],
             "relaxed_final_sample_alignment": 1-trace[-1]["relaxed_loss"],
             "optimized_discrete_retokenized_alignment": best["alignment_by_checkpoint"]["reference"],
             "random_discrete_retokenized_alignment": random_scores["reference"],
             "alignment_checkpoints": [c[0] for c in contexts],
             "optimized_alignment_by_checkpoint": best["alignment_by_checkpoint"],
             "random_alignment_by_checkpoint": random_scores,
             "selection_mean_alignment": best["discrete_alignment"],
             "review_token_count": sum(len(r["retokenized_review_ids"]) for r in optimized),
             "labels": {str(y): labels.count(y) for y in (0, 1)}, "target_id": target["id"],
             "steps": settings["steps"], "restarts": restarts,
             "poison_optimizer": method,
             "initialization": settings.get("initialization", "diffuse"),
             "selected_candidate": {k: v for k, v in best.items() if k != "rows"},
             "relaxed_final_sample_note": "Last pre-update relaxed sample of the last restart; not the selected text score",
             "discrete_candidates_evaluated": len(selection), **cost.finish()}
    stats["optimization_microbatch_passes"] = 2*math.ceil(n/settings["microbatch"])*settings["steps"]*restarts*len(contexts)
    stats["resumed_checkpoints"] = restored
    stats["optimization_microbatch_passes_this_invocation"] = 2*math.ceil(n/settings["microbatch"])*executed_steps*len(contexts)
    stats["duration_scope"] = "current invocation only; excludes previously saved computation"
    write_json(output / "alignment.json", stats)
    return stats
