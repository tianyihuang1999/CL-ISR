"""CSV dataset: text, label (0=real, 1=misleading), optional stance in {-1,0,1}."""

from __future__ import annotations

import hashlib

import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import two_views

STANCE_TO_ID = {-1: 0, 0: 1, 1: 2}
MISSING_STANCE = -100


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate annotations without inventing labels or converting NaN to text."""
    df = df.copy()
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: text, label")
    if df.empty:
        raise ValueError("Dataset must not be empty")
    if df["text"].isna().any() or df["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Text must be non-empty and non-missing")
    labels = pd.to_numeric(df["label"], errors="raise")
    if not labels.isin([0, 1]).all():
        raise ValueError("Labels must be 0 (real) or 1 (misleading)")
    df["label"] = labels.astype(int)
    if "stance" not in df:
        df["stance"] = float("nan")
    stance = pd.to_numeric(df["stance"], errors="raise")
    if not (stance.isna() | stance.isin(STANCE_TO_ID)).all():
        raise ValueError("Stance must be -1, 0, 1, or missing")
    df["stance"] = stance
    return df


class MisleadingTextDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        max_length: int = 512,
        augment_strategy: str = "hybrid",
        train: bool = True,
        use_cl: bool = True,
    ):
        self.df = validate_frame(pd.read_csv(path))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_strategy = augment_strategy
        self.train = train
        self.use_cl = use_cl

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
        stance = MISSING_STANCE if pd.isna(row["stance"]) else STANCE_TO_ID[int(row["stance"])]
        enc = self._encode(text)
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
            "stance": torch.tensor(stance, dtype=torch.long),
        }
        if self.train and self.use_cl:
            view_a, view_b = two_views(text, self.augment_strategy)
            for name, view in [("a", view_a), ("b", view_b)]:
                encoded = self._encode(view)
                item[f"view_{name}_ids"] = encoded["input_ids"].squeeze(0)
                item[f"view_{name}_mask"] = encoded["attention_mask"].squeeze(0)
        return item


def collate(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


class SimpleTokenizer:
    """Hash tokenizer for smoke tests without downloading BERT."""

    def __init__(self, vocab_size: int = 30522, max_length: int = 64):
        if vocab_size < 3 or max_length < 1:
            raise ValueError("vocab_size must be >= 3 and max_length must be positive")
        self.vocab_size = vocab_size
        self.model_max_length = max_length

    def __call__(self, text, truncation=True, padding="max_length", max_length=None, return_tensors="pt"):
        max_length = max_length or self.model_max_length
        # 0 is padding; 1 is reserved for an empty input. Stable across processes.
        ids = [(int.from_bytes(hashlib.sha256(tok.encode("utf-8")).digest()[:8], "big")
                % (self.vocab_size - 2)) + 2 for tok in text.split()][:max_length] or [1]
        pad = max_length - len(ids)
        input_ids = ids + [0] * pad
        mask = [1] * len(ids) + [0] * pad
        t_ids = torch.tensor([input_ids], dtype=torch.long)
        t_mask = torch.tensor([mask], dtype=torch.long)
        return {"input_ids": t_ids, "attention_mask": t_mask}
