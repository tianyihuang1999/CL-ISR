"""CSV dataset: text, label (0=real, 1=misleading), optional stance in {-1,0,1}."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import two_views

STANCE_TO_ID = {-1: 0, 0: 1, 1: 2}


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    view_ids: torch.Tensor
    view_mask: torch.Tensor
    labels: torch.Tensor
    stance: torch.Tensor


class MisleadingTextDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int = 512,
        augment_strategy: str = "hybrid",
        train: bool = True,
    ):
        self.df = pd.read_csv(path)
        if "text" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain columns: text, label")
        if "stance" not in self.df.columns:
            self.df["stance"] = 0
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_strategy = augment_strategy
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def _encode(self, text: str) -> dict:
        return self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        text = str(row["text"])
        label = int(row["label"])
        stance = STANCE_TO_ID.get(int(row["stance"]), 1)
        view_a, view_b = (two_views(text, self.augment_strategy) if self.train else (text, text))
        enc_a = self._encode(view_a)
        enc_b = self._encode(view_b)
        return {
            "input_ids": enc_a["input_ids"].squeeze(0),
            "attention_mask": enc_a["attention_mask"].squeeze(0),
            "view_ids": enc_b["input_ids"].squeeze(0),
            "view_mask": enc_b["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "stance": torch.tensor(stance, dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


class SimpleTokenizer:
    """Hash tokenizer for smoke tests without downloading BERT."""

    def __init__(self, vocab_size: int = 30522, max_length: int = 64):
        self.vocab_size = vocab_size
        self.model_max_length = max_length

    def __call__(self, text, truncation=True, padding="max_length", max_length=None, return_tensors="pt"):
        max_length = max_length or self.model_max_length
        ids = [(hash(tok) % (self.vocab_size - 2)) + 1 for tok in text.split()][:max_length]
        pad = max_length - len(ids)
        input_ids = ids + [0] * pad
        mask = [1] * len(ids) + [0] * pad
        t_ids = torch.tensor([input_ids], dtype=torch.long)
        t_mask = torch.tensor([mask], dtype=torch.long)
        return {"input_ids": t_ids, "attention_mask": t_mask}
