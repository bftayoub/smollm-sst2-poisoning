"""Two-pass exact microbatch gradient matching with replayed Gumbel noise.

Pass 1 measures the whole poison gradient. Its analytic cosine cotangent
is a constant for pass 2, which differentiates each poison gradient with
create_graph=True. This is the chain rule, not a detached surrogate.
"""
import math
import random

import torch
import torch.nn.functional as F

from .common import Cost, rng_state, seed_all, write_json
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


def discrete_alignment(model, codec, params, gt, rows, microbatch):
    total = [torch.zeros_like(g) for g in gt]
    for start in range(0, len(rows), microbatch):
        group = rows[start:start+microbatch]
        gp = gradient(model, params, codec.batch(group, next(model.parameters()).device))
        with torch.no_grad():
            for acc, g in zip(total, gp):
                acc.add_(g, alpha=len(group)/len(rows))
        del gp
    value = float(cosine(gt, total))
    if not math.isfinite(value):
        raise RuntimeError("Non-finite discrete alignment")
    return value


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


def optimize(model, codec, target, cfg, output, all_targets):
    output.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["poison_seed"], cfg["threads"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    device = next(model.parameters()).device
    cost = Cost(device)
    names, params = parameters_for_alignment(model, cfg["poison"]["alignment_parameters"])
    target_row = {**target, "label": target["desired_label"]}
    gt = tuple(g.detach() for g in gradient(model, params, codec.batch([target_row], device)))
    vocab_list = safe_vocab(codec.tokenizer, cfg["poison"].get("vocab_limit"))
    vocab = torch.tensor(vocab_list, device=device)
    n, length = cfg["poison"]["count"], cfg["poison"]["length"]
    labels = [i % 2 for i in range(n)]
    z = torch.nn.Parameter(torch.randn(n, length, len(vocab), device=device)*0.01)
    optimizer = torch.optim.Adam([z], lr=cfg["poison"]["learning_rate"])
    initial = z.detach().argmax(dim=-1)
    write_json(output / "initial_tokens.json", vocab[initial].cpu().tolist())
    trace = []
    for step in range(cfg["poison"]["steps"]):
        fraction = step/max(1, cfg["poison"]["steps"]-1)
        temperature = cfg["poison"]["temperature_start"] * (
            cfg["poison"]["temperature_end"]/cfg["poison"]["temperature_start"])**fraction
        before = z.detach().clone()
        value, dz = matching_step(model, codec, params, gt, z, vocab, labels,
                                  temperature, cfg["poison"]["microbatch"])
        optimizer.zero_grad(set_to_none=True)
        z.grad = dz
        optimizer.step()
        delta = float((z.detach()-before).abs().max())
        if delta == 0 or not torch.isfinite(z).all():
            raise RuntimeError("Poison distribution did not update finitely")
        trace.append({"step": step+1, "temperature": temperature, "relaxed_loss": value,
                      "gradient_norm": float(dz.norm()), "max_logit_update": delta})
        write_json(output / "optimization_trace.json", trace)
        print(f"poison step={step+1} alignment={1-value:.6f} delta={delta:.6f}", flush=True)
        if (step+1) % cfg["poison"]["save_every"] == 0:
            torch.save({"logits": z.detach().cpu(), "optimizer": optimizer.state_dict(),
                        "rng": rng_state(), "step": step+1}, output / "poison_state.pt")
    # Victims receive these serialized texts, never soft embeddings or original IDs.
    ids = vocab[z.detach().argmax(dim=-1)].cpu().tolist()
    optimized = export_rows(codec, ids, labels, all_targets)
    random_rows = matched_random(codec, optimized, vocab_list, cfg["poison_seed"]+1, all_targets)
    write_json(output / "optimized.json", optimized)
    write_json(output / "random.json", random_rows)
    stats = {"alignment_parameter_names": names,
             "alignment_parameter_count": sum(p.numel() for p in params),
             "alignment_parameters": cfg["poison"]["alignment_parameters"],
             "ablation": cfg["poison"]["alignment_parameters"] != "all",
             "usable_vocabulary_size": len(vocab_list), "poison_seed": cfg["poison_seed"],
             "relaxed_final_sample_alignment": 1-trace[-1]["relaxed_loss"],
             "optimized_discrete_retokenized_alignment": discrete_alignment(model, codec, params, gt, optimized, cfg["poison"]["microbatch"]),
             "random_discrete_retokenized_alignment": discrete_alignment(model, codec, params, gt, random_rows, cfg["poison"]["microbatch"]),
             "review_token_count": sum(len(r["retokenized_review_ids"]) for r in optimized),
             "labels": {str(y): labels.count(y) for y in (0, 1)}, "target_id": target["id"],
             "steps": cfg["poison"]["steps"], **cost.finish()}
    stats["optimization_microbatch_passes"] = 2*math.ceil(n/cfg["poison"]["microbatch"])*cfg["poison"]["steps"]
    write_json(output / "alignment.json", stats)
    torch.save({"logits": z.detach().cpu(), "optimizer": optimizer.state_dict(), "vocab": vocab.cpu(),
                "rng": rng_state(), "step": len(trace)}, output / "poison_state.pt")
    return stats
