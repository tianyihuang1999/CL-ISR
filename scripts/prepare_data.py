"""Convert public rumor/fake-news CSVs into CL-ISR format.

Expected input columns can be remapped. Missing stances remain unannotated.
PHEME often includes rumor vs non-rumor plus stance
annotations on replies; FakeNewsNet typically provides fake/real news labels.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cl_isr.data import validate_frame


def clean_text(text: str, stopwords: set[str]) -> str:
    """Explicit, reproducible cleaning approximation; the paper supplies no lists."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = "".join(c if c.isspace() or unicodedata.category(c)[0] in {"L", "N"} else " " for c in text)
    return " ".join(t for t in text.split() if t.casefold() not in stopwords)


def prepare(src: str, dest_dir: str, text_col: str, label_col: str, stance_col: str | None,
            seed: int = 42, clean: bool = False, stopwords_file: str | None = None) -> None:
    df = pd.read_csv(src)
    out = pd.DataFrame(
        {
            "text": df[text_col],
            "label": df[label_col],
            "stance": df[stance_col] if stance_col and stance_col in df.columns else float("nan"),
        }
    )
    out = out.dropna(subset=["text", "label"])
    out["text"] = out["text"].astype(str).str.strip()
    if stopwords_file and not clean:
        raise ValueError("--stopwords-file requires --clean-text")
    if clean:
        stopwords = set()
        if stopwords_file:
            stopwords = {s.strip().casefold() for s in Path(stopwords_file).read_text(encoding="utf-8").splitlines() if s.strip()}
        out["text"] = out["text"].map(lambda t: clean_text(t, stopwords))
    out = out.loc[out["text"].ne("")]
    out = validate_frame(out)
    # Exact duplicates must not leak across the train/validation/test splits.
    if out.groupby("text")[["label", "stance"]].nunique().gt(1).any().any():
        raise ValueError("Duplicate texts have conflicting labels or stances")
    out = out.sort_values("stance", na_position="last").drop_duplicates("text").sort_index()
    train, rest = train_test_split(out, test_size=0.2, random_state=seed, stratify=out["label"])
    val, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest["label"])
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    train.to_csv(dest / "train.csv", index=False)
    val.to_csv(dest / "val.csv", index=False)
    test.to_csv(dest / "test.csv", index=False)
    print(f"wrote {len(train)}/{len(val)}/{len(test)} to {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--stance-col", default="stance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-text", action="store_true")
    parser.add_argument("--stopwords-file", help="UTF-8 file containing one stopword per line")
    args = parser.parse_args()
    stance = args.stance_col if args.stance_col else None
    prepare(args.src, args.dest, args.text_col, args.label_col, stance, args.seed, args.clean_text, args.stopwords_file)


if __name__ == "__main__":
    main()
