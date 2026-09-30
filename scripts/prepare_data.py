"""Convert public rumor/fake-news CSVs into CL-ISR format.

Expected input columns can be remapped. Stance is optional; missing values
default to 0 (neutral). PHEME often includes rumor vs non-rumor plus stance
annotations on replies; FakeNewsNet typically provides fake/real news labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare(src: str, dest_dir: str, text_col: str, label_col: str, stance_col: str | None, seed: int = 42) -> None:
    df = pd.read_csv(src)
    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": df[label_col].astype(int),
            "stance": df[stance_col].astype(int) if stance_col and stance_col in df.columns else 0,
        }
    )
    out = out.dropna(subset=["text", "label"])
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
    args = parser.parse_args()
    stance = args.stance_col if args.stance_col else None
    prepare(args.src, args.dest, args.text_col, args.label_col, stance)


if __name__ == "__main__":
    main()
