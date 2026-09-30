from pathlib import Path
import sys

import torch

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
