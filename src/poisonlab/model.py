import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import device_for, file_digest, read_json, write_json


class LabelCodec:
    def __init__(self, tokenizer, max_length=128, codes=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        tokenizer.pad_token = tokenizer.eos_token
        self.codes = codes
        if codes is None:
            for pair in ((" 0", " 1"), (" A", " B"), (" negative", " positive")):
                try:
                    self.codes = list(pair)
                    self.label_ids = self.verify("Review: test\nSentiment:")
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("No stable single-token label codes")
        self.label_ids = self.verify("Review: test\nSentiment:")

    def verify(self, prompt):
        base = self.tokenizer.encode(prompt, add_special_tokens=False)
        result = []
        for code in self.codes:
            full = self.tokenizer.encode(prompt + code, add_special_tokens=False)
            if full[:-1] != base or len(full) != len(base) + 1:
                raise ValueError(f"Label {code!r} is not one stable continuation token")
            result.append(full[-1])
        if len(set(result)) != 2:
            raise ValueError("Label tokens must differ")
        return result

    def prompt_ids(self, text):
        tok = self.tokenizer
        review = tok.encode(text, add_special_tokens=False)
        # Preserve exact English suffix and reserve one token for the label.
        while True:
            prompt = "Review: " + text + "\nSentiment:"
            ids = tok.encode(prompt, add_special_tokens=False)
            if len(ids) <= self.max_length - 1:
                if self.verify(prompt) != self.label_ids:
                    raise ValueError("Context-dependent label token changed")
                return ids
            excess = len(ids) - self.max_length + 1
            review = review[:max(0, len(review) - max(1, excess))]
            text = tok.decode(review, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if not review and len(ids) > self.max_length - 1:
                empty = tok.encode("Review: \nSentiment:", add_special_tokens=False)
                if len(empty) > self.max_length - 1:
                    raise ValueError("max_length too small for prompt")

    def batch(self, rows, device, labeled=True):
        seqs, labels = [], []
        for row in rows:
            ids = self.prompt_ids(row["text"])
            lab = [-100] * len(ids)
            if labeled:
                ids = ids + [self.label_ids[row["label"]]]
                lab = lab + [ids[-1]]
            seqs.append(ids)
            labels.append(lab)
        n = max(map(len, seqs))
        batch = {"input_ids": torch.tensor([s + [self.tokenizer.pad_token_id] * (n-len(s)) for s in seqs], device=device),
                 "attention_mask": torch.tensor([[1]*len(s)+[0]*(n-len(s)) for s in seqs], device=device)}
        if labeled:
            batch["labels"] = torch.tensor([s+[-100]*(n-len(s)) for s in labels], device=device)
        return batch

    def metadata(self):
        return {"codes": self.codes, "label_ids": self.label_ids,
                "max_length": self.max_length, "loss": "full-vocabulary label-token CE"}


def prepare_initial(cfg, root):
    local = cfg["model"].get("local_checkpoint")
    if local:
        from pathlib import Path
        source = Path(local).resolve()
        revision = "local:" + file_digest(source / "model.safetensors")
        kwargs = {"local_files_only": True}
    else:
        from huggingface_hub import HfApi
        source = cfg["model"]["name"]
        revision = HfApi().model_info(source, revision=cfg["model"]["revision"]).sha
        kwargs = {"revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs,
                                               attn_implementation="eager", torch_dtype=torch.float32)
    model.config.use_cache = False
    path = root / "initial"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    codec = LabelCodec(tokenizer, cfg["max_length"])
    write_json(root / "model_manifest.json", {"model": cfg["model"]["name"], "revision": revision,
               "local_checkpoint": str(source) if local else None,
               "parameter_count": sum(p.numel() for p in model.parameters()), "dtype": "float32",
               "attention": "eager", "labels": codec.metadata(),
               "initial_weights_sha256": file_digest(path / "model.safetensors")})


def load_model(cfg, root, checkpoint="initial"):
    path = root / checkpoint
    manifest = read_json(root / "model_manifest.json")
    if checkpoint == "initial" and "initial_weights_sha256" in manifest:
        if file_digest(path / "model.safetensors") != manifest["initial_weights_sha256"]:
            raise ValueError("Initial checkpoint weights changed")
    tokenizer = AutoTokenizer.from_pretrained(root / "initial", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                                attn_implementation="eager", torch_dtype=torch.float32)
    model.config.use_cache = False
    model.to(device_for(cfg))
    metadata = manifest["labels"]
    codec = LabelCodec(tokenizer, cfg["max_length"], metadata["codes"])
    if codec.label_ids != metadata["label_ids"]:
        raise ValueError("Label tokenizer changed")
    return model, codec
