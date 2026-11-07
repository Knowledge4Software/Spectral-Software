"""Fine-tuned text baselines with the same resumable selection/test protocol.

C4-inspired adaptation: CodeBERT CLS vectors and task-masked in-batch contrastive
learning. We mask false negatives rather than repeat other tasks as upstream
does; both pair directions and all known training endpoints can supply anchors.
This is an adaptation, not a reproduction of the original C4 experiment.
Sources: https://github.com/Chenning-Tao/C4 (model.py and run_con.py).
Qwen pooling: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
LoRA: https://huggingface.co/docs/peft/en/package_reference/lora
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .io_utils import replace_file
from .protocol import fingerprint
from .training import SessionBudgetReached


@dataclass(frozen=True)
class TransformerConfig:
    model_id: str = "microsoft/codebert-base"
    revision: str | None = None
    objective: str = "pair_bce"
    pooling: str = "cls"
    max_length: int = 512
    lora_rank: int = 0
    contrastive_weight: float = .1
    gradient_checkpointing: bool = True
    family: str = "transformer_pair"


@dataclass
class TokenBatch:
    inputs: dict
    tasks: torch.Tensor

    def to(self, device):
        return TokenBatch({key: value.to(device) for key, value in self.inputs.items()}, self.tasks.to(device))


class _Pairs(Dataset):
    def __init__(self, frame, index):
        self.rows = [(index[str(a)], index[str(b)], float(y))
                     for a, b, y in frame[["left_id", "right_id", "label"]].itertuples(index=False, name=None)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, row):
        return self.rows[row]


class TokenCorpus:
    def __init__(self, tokens, masks, ids, tasks):
        self.tokens, self.masks = tokens, masks
        self.index = {key: i for i, key in enumerate(ids)}
        self.tasks = np.asarray(tasks, dtype=np.int64)

    def _batch(self, rows):
        rows = np.asarray(rows)
        # Fresh writable arrays avoid mmap warnings and never alter the cache.
        return TokenBatch({"input_ids": torch.tensor(np.asarray(self.tokens[rows]), dtype=torch.long),
                           "attention_mask": torch.tensor(np.asarray(self.masks[rows]), dtype=torch.long)},
                          torch.tensor(self.tasks[rows]))

    def pair_loader(self, frame, config, batches=None):
        def collate(rows):
            a, b, y = zip(*rows)
            return self._batch(a), self._batch(b), torch.tensor(y, dtype=torch.float32)
        kwargs = {"batch_sampler": batches} if batches is not None else {"batch_size": config.batch_size}
        return DataLoader(_Pairs(frame, self.index), collate_fn=collate, num_workers=0,
                          generator=torch.Generator().manual_seed(config.seed + 9187), **kwargs)


def task_contrastive(vectors, tasks, scale=5.0):
    """Supervised contrastive loss; no self positives or same-task negatives."""
    vectors = F.normalize(vectors.float(), dim=-1)
    scores = scale * vectors @ vectors.T
    diagonal = torch.eye(len(tasks), device=tasks.device, dtype=torch.bool)
    known = tasks.ge(0)
    allowed = known[:, None] & known[None, :] & ~diagonal
    positive = tasks[:, None].eq(tasks[None, :]) & allowed
    anchors = positive.any(1) & (allowed & ~positive).any(1)
    if not anchors.any():
        return vectors.sum() * 0
    logits = scores[anchors].masked_fill(~allowed[anchors], -torch.inf)
    log_probability = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    positive = positive[anchors]
    selected = log_probability.masked_fill(~positive, 0)
    return -(selected.sum(1) / positive.sum(1)).mean()


class TransformerPairModel(nn.Module):
    def __init__(self, config, *, encoder=None):
        super().__init__()
        if config.objective not in {"pair_bce", "c4_adapted", "pair_bce_contrastive"}:
            raise ValueError("Unknown transformer objective")
        if config.pooling not in {"cls", "last_token"}:
            raise ValueError("Pooling must be explicit")
        self.settings = config
        if encoder is None:
            from transformers import AutoModel
            if not config.revision:
                raise ValueError("Resolve and pin the model commit before constructing the baseline")
            encoder = AutoModel.from_pretrained(config.model_id, revision=config.revision,
                                                trust_remote_code=False, torch_dtype=torch.float32)
            if hasattr(encoder.config, "use_cache"):
                encoder.config.use_cache = False
            if config.lora_rank:
                from peft import LoraConfig, get_peft_model
                encoder = get_peft_model(encoder, LoraConfig(
                    r=config.lora_rank, lora_alpha=2 * config.lora_rank, lora_dropout=.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none"))
                encoder.enable_input_require_grads()
            if config.gradient_checkpointing:
                encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.encoder = encoder
        dimension = encoder.config.hidden_size
        if config.objective != "c4_adapted":
            self.classifier = nn.Sequential(nn.Linear(2 * dimension + 2, 256), nn.GELU(),
                                            nn.Dropout(.1), nn.Linear(256, 1))

    def encode(self, batch):
        hidden = self.encoder(**batch.inputs).last_hidden_state
        if self.settings.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
            last = positions.masked_fill(~batch.inputs["attention_mask"].bool(), -1).max(1).values
            pooled = hidden[torch.arange(len(hidden), device=hidden.device), last]
        return F.normalize(pooled.float(), dim=-1)

    def forward(self, left, right, temperature=None):
        a, b = self.encode(left), self.encode(right)
        cosine = (a * b).sum(-1)
        if self.settings.objective == "c4_adapted":
            logits = 5.0 * cosine
        else:
            features = torch.cat(((a - b).abs(), a * b, cosine[:, None], (a - b).norm(dim=-1, keepdim=True)), -1)
            logits = self.classifier(features).squeeze(-1)
        return logits, {"cosine": cosine, "vectors": torch.cat((a, b)),
                        "tasks": torch.cat((left.tasks, right.tasks))}

    def compute_loss(self, logits, auxiliary, labels, *, positive_class_weight=1.0, **unused):
        if self.settings.objective == "c4_adapted":
            loss = task_contrastive(auxiliary["vectors"], auxiliary["tasks"])
            return loss, {"c4_adapted_contrastive": loss.detach()}
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float(),
                                                  pos_weight=logits.new_tensor(positive_class_weight))
        if self.settings.objective == "pair_bce_contrastive":
            cosine = auxiliary["cosine"]
            contrastive = (labels * (1 - cosine).square() + (1 - labels) * F.relu(cosine - .2).square()).mean()
            loss = loss + self.settings.contrastive_weight * contrastive
        return loss, {"training_loss": loss.detach()}


def prepare_tokens(corpus, frames, settings, output, *, clock, resume=None):
    """Resumable, commit-pinned tokenization; never retains model activations."""
    from transformers import AutoConfig, AutoTokenizer
    from .data import pair_code_ids
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "tokens.json"
    if resume and not manifest_path.exists():
        for name in ("tokens.json", "tokens.npy", "masks.npy"):
            shutil.copyfile(Path(resume) / name, output / name)
    state = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    clock.check()
    revision = state["revision"] if state else settings.revision
    resolved = AutoConfig.from_pretrained(settings.model_id, revision=revision, trust_remote_code=False)._commit_hash
    settings = replace(settings, revision=resolved)
    ids = sorted(pair_code_ids(*frames.values()))
    identity = fingerprint({"model": settings.model_id, "revision": resolved, "length": settings.max_length,
                            "whitespace_normalized": settings.objective == "c4_adapted",
                            "codes": [(key, hashlib.sha256(corpus.code_texts[key].encode()).hexdigest()) for key in ids]})
    if state and state["identity"] != identity:
        raise ValueError("Token-cache identity mismatch")
    tokenizer = AutoTokenizer.from_pretrained(settings.model_id, revision=resolved, trust_remote_code=False,
                                              padding_side="left" if settings.pooling == "last_token" else "right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if state is None:
        np.lib.format.open_memmap(output / "tokens.npy", mode="w+", dtype=np.int32,
                                  shape=(len(ids), settings.max_length)).flush()
        np.lib.format.open_memmap(output / "masks.npy", mode="w+", dtype=np.uint8,
                                  shape=(len(ids), settings.max_length)).flush()
        state = {"identity": identity, "revision": resolved, "next_index": 0, "complete": False,
                 "codes": len(ids), "truncated_codes": 0, "max_length": settings.max_length}
    def save():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        replace_file(temporary, manifest_path)
    tokens = np.load(output / "tokens.npy", mmap_mode="r+")
    masks = np.load(output / "masks.npy", mmap_mode="r+")
    try:
        while state["next_index"] < len(ids):
            clock.check()
            start, stop = state["next_index"], min(state["next_index"] + 256, len(ids))
            texts = [corpus.code_texts[key] for key in ids[start:stop]]
            if settings.objective == "c4_adapted":
                texts = [" ".join(text.split()) for text in texts]
            lengths = [len(row) for row in tokenizer(texts, truncation=False)["input_ids"]]
            batch = tokenizer(texts, padding="max_length", truncation=True, max_length=settings.max_length)
            tokens[start:stop] = batch["input_ids"]
            masks[start:stop] = batch["attention_mask"]
            tokens.flush(); masks.flush()
            state.update(next_index=stop, complete=stop == len(ids))
            state["truncated_codes"] += sum(length > settings.max_length for length in lengths)
            save()
    except SessionBudgetReached:
        save()
        return None, settings, state
    tasks = {key: corpus.code_metadata.get(key, {}).get("problem_id") for key in ids}
    train_ids = pair_code_ids(frames["train"])
    vocabulary = {key: i for i, key in enumerate(sorted({str(tasks[key]) for key in train_ids if tasks[key] is not None}))}
    task_ids = [vocabulary.get(str(tasks[key]), -1) if key in train_ids else -1 for key in ids]
    if settings.objective == "c4_adapted" and any(task < 0 for key, task in zip(ids, task_ids) if key in train_ids):
        raise ValueError("C4 adaptation requires known training problem IDs; do not infer them from test labels")
    return TokenCorpus(tokens, masks, ids, task_ids), settings, state
