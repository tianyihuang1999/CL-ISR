import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import torch

from cl_isr.data import MisleadingTextDataset, SimpleTokenizer, validate_frame
from cl_isr.losses import stance_loss
from scripts.prepare_data import prepare


def test_tokenizer_is_stable_across_hash_seeds():
    code = "from cl_isr.data import SimpleTokenizer; import json; print(json.dumps(SimpleTokenizer()('some real news')['input_ids'].tolist()))"
    outputs = []
    for seed in ("1", "987"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        outputs.append(subprocess.check_output([sys.executable, "-c", code], env=env, text=True))
    assert json.loads(outputs[0]) == json.loads(outputs[1])


def test_original_text_and_missing_stance(tmp_path, monkeypatch):
    path = tmp_path / "data.csv"
    pd.DataFrame({"text": ["original text", "other text"], "label": [0, 1], "stance": [None, 0]}).to_csv(path, index=False)
    monkeypatch.setattr("cl_isr.data.two_views", lambda *args: ("augmented first", "augmented second"))
    tokenizer = SimpleTokenizer(max_length=7)
    ds = MisleadingTextDataset(path, tokenizer, max_length=7)
    item = ds[0]
    assert item["stance"].item() == -100
    assert ds[1]["stance"].item() == 1  # Actual neutral annotation remains supervised.
    assert torch.equal(item["input_ids"], tokenizer("original text")["input_ids"].squeeze(0))
    assert not torch.equal(item["input_ids"], item["view_a_ids"])
    assert "view_a_ids" not in MisleadingTextDataset(path, tokenizer, train=False)[0]
    assert "view_a_ids" not in MisleadingTextDataset(path, tokenizer, use_cl=False)[0]


@pytest.mark.parametrize("row", [
    {"text": None, "label": 0}, {"text": " ", "label": 0},
    {"text": "ok", "label": 0.5}, {"text": "ok", "label": None},
    {"text": "ok", "label": 2}, {"text": "ok", "label": 0, "stance": 5},
])
def test_invalid_annotations_rejected(row):
    with pytest.raises(ValueError):
        validate_frame(pd.DataFrame([row]))


def test_unlabeled_stance_loss_is_finite_and_differentiable():
    logits = torch.randn(3, 3, requires_grad=True)
    loss = stance_loss(logits, torch.full((3,), -100))
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    labels = torch.tensor([-100, 1, 2])
    assert torch.allclose(stance_loss(logits, labels), torch.nn.functional.cross_entropy(logits[1:], labels[1:]))


def test_prepare_drops_missing_and_duplicates_before_split(tmp_path):
    rows = [{"text": f"text {i}", "label": i % 2} for i in range(40)]
    rows += [rows[0], {"text": None, "label": 1}, {"text": "bad", "label": None}]
    path = tmp_path / "raw.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = tmp_path / "splits"
    prepare(str(path), str(out), "text", "label", "stance")
    splits = [pd.read_csv(out / f"{name}.csv") for name in ["train", "val", "test"]]
    assert [len(s) for s in splits] == [32, 4, 4]
    combined = pd.concat(splits)
    assert combined["text"].nunique() == 40
    assert combined["stance"].isna().all()


def test_prepare_rejects_conflicting_duplicates(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame({"text": ["same", "same"], "label": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="conflicting"):
        prepare(str(path), str(tmp_path / "out"), "text", "label", None)
