import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_isr.data import MisleadingTextDataset, collate
from cl_isr.model import CLISR, CLISRConfig
from cl_isr.train import build_tokenizer, evaluate, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default="outputs/best.pt")
    parser.add_argument("--split", default="test_path")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(cfg)
    max_len = 64 if not cfg.get("pretrained", True) else int(cfg["max_length"])
    ds = MisleadingTextDataset(cfg[args.split], tokenizer, max_len, train=False)
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), collate_fn=collate)
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)
    model = CLISR(CLISRConfig(**{k: v for k, v in ckpt["config"].items() if k in CLISRConfig.__dataclass_fields__}))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(evaluate(model, loader, device))


if __name__ == "__main__":
    main()
