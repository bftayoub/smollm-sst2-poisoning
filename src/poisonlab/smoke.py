from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from .common import digest, seed_all, write_json
from .data import partition
from .experiment import bind_run, execute_stage, reference_stage, run
from .model import LabelCodec


def tiny_components():
    words = ["<unk>", "<eos>", "Review", ":", "Sentiment", "0", "1", "good", "bad", "movie"]
    words += [f"item{i}" for i in range(80)]
    vocab = {w: i for i, w in enumerate(words)}
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", eos_token="<eos>")
    config = LlamaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                         max_position_embeddings=128, tie_word_embeddings=True,
                         attention_dropout=0.0, bos_token_id=None, eos_token_id=1, pad_token_id=1)
    config._attn_implementation = "eager"
    config.use_cache = False
    return LlamaForCausalLM(config), tokenizer


def smoke(root):
    cfg = {"run_name": "smoke", "device": "cpu", "threads": 1, "max_length": 48,
           "target_count": 1, "balanced_targets": False, "target_confidence": 0.0,
           "evaluation_batch_size": 4, "split_seed": 1729, "reference_seed": 42,
           "poison_seed": 31415, "victim_seeds": [101],
           "model": {"name": "tiny-random-llama", "revision": "local-fixture-v1"},
           "data": {"name": "synthetic-fixture", "revision": "local-fixture-v1",
                    "train_size": 16, "dev_size": 4, "reserve_size": 4},
           "train": {"epochs": 2, "batch_size": 4, "learning_rate": 0.005,
                     "weight_decay": 0.01, "max_grad_norm": 1.0},
           "poison": {"count": 2, "length": 3, "steps": 2, "microbatch": 1,
                      "save_every": 1, "learning_rate": 0.3, "temperature_start": 1.0,
                      "temperature_end": 0.3, "alignment_parameters": "all", "vocab_limit": None}}
    bind_run(cfg, root)
    seed_all(cfg["reference_seed"], 1)
    model, tokenizer = tiny_components()
    codec = LabelCodec(tokenizer, cfg["max_length"])
    model.save_pretrained(root / "initial", safe_serialization=True)
    tokenizer.save_pretrained(root / "initial")
    write_json(root / "model_manifest.json", {"model": "tiny-random-llama", "revision": "local-fixture-v1",
               "labels": codec.metadata(), "parameter_count": sum(p.numel() for p in model.parameters())})
    train = [{"sentence": f"{'good' if i%2 else 'bad'} movie item{i}", "label": i%2} for i in range(20)]
    val = [{"sentence": f"{'good' if i%2 else 'bad'} movie item{i}", "label": i%2} for i in range(20, 36)]
    parts = partition(train, val, cfg)
    write_json(root / "partitions.json", parts)
    write_json(root / "data_manifest.json", {"dataset": "synthetic-fixture", "revision": "local-fixture-v1",
               "actual_sizes": {k: len(v) for k, v in parts.items()}, "partition_sha256": digest(parts)})
    execute_stage(root, "reference", lambda: reference_stage(cfg, root))
    run(cfg, root, "poisons")
    run(cfg, root, "victims")
    write_json(root / "smoke_report.json", {"status": "passed", "technical_only": True,
               "model": "tiny random Llama, not pretrained SmolLM", "dataset": "synthetic fixture, not SST-2",
               "checks": ["label-only training", "second-order eager autograd", "nonzero poison update",
                          "decode/retokenize alignment", "all three victims from identical initial checkpoint",
                          "target exclusion", "paired update/exposure/order/token-length budgets"],
               "experimental_efficacy_demonstrated": False})
    print("SMOKE PASSED (technical verification only)", flush=True)
