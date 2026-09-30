from pathlib import Path
import sys

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_isr.augment import two_views
from cl_isr.losses import infonce_loss, total_loss
from cl_isr.model import CLISR, CLISRConfig


def test_infonce_shape():
    z1 = torch.randn(4, 8)
    z2 = torch.randn(4, 8)
    loss = infonce_loss(z1, z2, 0.07)
    assert loss.ndim == 0


def test_two_views_not_empty():
    a, b = two_views("officials confirm the report is accurate", "hybrid")
    assert a and b


def test_forward_dummy():
    cfg = CLISRConfig(pretrained=False, hidden_size=32, lstm_hidden=16, vocab_size=100)
    model = CLISR(cfg)
    ids = torch.randint(1, 100, (2, 12))
    mask = torch.ones(2, 12, dtype=torch.long)
    out = model(ids, mask, ids, mask)
    assert out["logits"].shape == (2, 2)
    assert out["stance_logits"].shape == (2, 3)
    loss = total_loss(
        infonce_loss(out["h"], out["h2"]),
        torch.nn.functional.cross_entropy(out["stance_logits"], torch.tensor([1, 0])),
        torch.nn.functional.cross_entropy(out["logits"], torch.tensor([0, 1])),
        torch.tensor(0.0),
        1.0,
        1.0,
        1.0,
    )
    loss.backward()


def test_infonce_prefers_correct_pairs():
    z = torch.eye(4)
    assert infonce_loss(z, z) < infonce_loss(z, z.roll(1, 0))
    for temperature in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            infonce_loss(z, z, temperature)


def test_isr_removal_and_mean_fusion():
    ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    mask = ids.ne(0).long()
    model = CLISR(CLISRConfig(pretrained=False, vocab_size=10, hidden_size=8, lstm_hidden=4, use_isr=False))
    out = model(ids, mask)
    assert model.isr is None and model.fusion is None
    assert out["stance_logits"] is None
    assert torch.equal(out["r"], out["h"])
    model = CLISR(CLISRConfig(pretrained=False, vocab_size=10, hidden_size=8, lstm_hidden=4, fusion="mean"))
    out = model(ids, mask)
    assert model.fusion is None
    assert torch.equal(out["r"], (out["h"] + out["s"]) / 2)
    assert out["alpha"][0, 2].item() == 0
    assert torch.allclose(out["alpha"].sum(-1), torch.ones(2))


def test_isr_padding_does_not_change_prediction():
    model = CLISR(CLISRConfig(pretrained=False, vocab_size=10, hidden_size=8, lstm_hidden=4)).eval()
    with torch.no_grad():
        a = model(torch.tensor([[1, 2]]), torch.tensor([[1, 1]]))
        b = model(torch.tensor([[1, 2, 0, 0]]), torch.tensor([[1, 1, 0, 0]]))
    assert torch.allclose(a["logits"], b["logits"], atol=1e-6)
