import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cl_isr.data import MisleadingTextDataset, collate
from cl_isr.train import checkpoint_tokenizer, evaluate, load_checkpoint, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt", default="outputs/best.pt")
    parser.add_argument("--split", default="test_path")
    parser.add_argument("--data", help="Override the evaluation CSV (including cross-domain evaluation)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.ckpt, device)
    tokenizer = checkpoint_tokenizer(args.ckpt, ckpt)
    max_len = ckpt["config"]["max_length"]
    ds = MisleadingTextDataset(args.data or cfg[args.split], tokenizer, max_len, train=False)
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), collate_fn=collate)
    print(evaluate(model, loader, device))


if __name__ == "__main__":
    main()
