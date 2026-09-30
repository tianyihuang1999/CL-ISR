import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader

from cl_isr.data import MisleadingTextDataset, collate
from cl_isr.train import checkpoint_tokenizer, evaluate, load_checkpoint, load_config, train


ROOT = Path(__file__).resolve().parents[1]


def smoke_config(tmp_path):
    cfg = load_config(ROOT / "configs/smoke.yaml")
    cfg.update(hidden_size=8, lstm_hidden=4, vocab_size=128, max_length=11,
               output_dir=str(tmp_path / "run"), epochs=1)
    for split in ["train", "val", "test"]:
        cfg[f"{split}_path"] = str(ROOT / f"data/sample/{split}.csv")
    return cfg


def test_training_checkpoint_and_fresh_process_evaluation(tmp_path):
    cfg = smoke_config(tmp_path)
    expected = train(cfg)
    ckpt_path = Path(cfg["output_dir"]) / "best.pt"
    model, ckpt = load_checkpoint(ckpt_path)
    assert ckpt["config"]["max_length"] == 11
    tokenizer = checkpoint_tokenizer(ckpt_path, ckpt)
    ds = MisleadingTextDataset(cfg["test_path"], tokenizer, 11, train=False)
    assert evaluate(model, DataLoader(ds, batch_size=4, collate_fn=collate), "cpu") == expected
    # Intentionally pass default.yaml (BERT, max_length=512): model/tokenizer settings
    # must come from the checkpoint. This also exercises cross-process hashing.
    output = subprocess.check_output([sys.executable, str(ROOT / "scripts/evaluate.py"),
                                     "--config", str(ROOT / "configs/default.yaml"),
                                     "--ckpt", str(ckpt_path), "--data", cfg["test_path"]], cwd=ROOT, text=True)
    import ast
    actual = ast.literal_eval(output.strip())
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], abs=1e-6)


def test_early_stopping_tracks_loss_not_f1(tmp_path, monkeypatch):
    cfg = smoke_config(tmp_path)
    cfg.update(epochs=5, patience=1)
    scores = iter([(0.8, 0.9), (0.7, 0.8), (0.75, 1.0), (0.7, 0.8)])

    def fake_evaluate(*args):
        loss, f1 = next(scores)
        return {"loss": loss, "f1": f1, "accuracy": f1, "recall": f1}

    monkeypatch.setattr("cl_isr.train.evaluate", fake_evaluate)
    train(cfg)
    report = json.loads((Path(cfg["output_dir"]) / "history.json").read_text())
    assert len(report["history"]) == 3
    assert report["best_epoch"] == 2
    _, ckpt = load_checkpoint(Path(cfg["output_dir"]) / "best.pt")
    assert ckpt["epoch"] == 2


def test_legacy_checkpoint_fails_with_migration_message(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"config": {}, "model": {}}, path)
    with pytest.raises(ValueError, match="retrain"):
        load_checkpoint(path)


@pytest.mark.parametrize("settings", [{"use_cl": False}, {"use_isr": False}, {"fusion": "mean"}])
def test_ablation_training(tmp_path, settings):
    cfg = {**smoke_config(tmp_path), **settings}
    result = train(cfg)
    assert 0 <= result["f1"] <= 1
