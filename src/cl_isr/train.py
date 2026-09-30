"""Joint training of CL-ISR (Eq. 16) with warmup + cosine schedule and early stopping."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MisleadingTextDataset, SimpleTokenizer, collate
from .losses import infonce_loss, l2_regularization, total_loss
from .metrics import compute_metrics
from .model import CLISR, CLISRConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tokenizer(cfg: dict):
    if not cfg.get("pretrained", True):
        return SimpleTokenizer(max_length=min(64, int(cfg["max_length"])))
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg["encoder_name"])


def evaluate(model, loader, device) -> dict:
    model.eval()
    ys, ps = [], []
    total_loss_val = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            loss = F.cross_entropy(out["logits"], batch["labels"])
            total_loss_val += loss.item() * batch["labels"].size(0)
            n += batch["labels"].size(0)
            ys.extend(batch["labels"].cpu().tolist())
            ps.extend(out["logits"].argmax(-1).cpu().tolist())
    metrics = compute_metrics(ys, ps)
    metrics["loss"] = total_loss_val / max(n, 1)
    return metrics


def train(cfg: dict) -> dict:
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = 64 if not cfg.get("pretrained", True) else int(cfg["max_length"])
    tokenizer = build_tokenizer(cfg)
    train_ds = MisleadingTextDataset(
        cfg["train_path"], tokenizer, max_len, cfg.get("augmentation", "hybrid"), train=True
    )
    val_ds = MisleadingTextDataset(cfg["val_path"], tokenizer, max_len, train=False)
    test_ds = MisleadingTextDataset(cfg["test_path"], tokenizer, max_len, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate)

    model_cfg = CLISRConfig(
        encoder_name=cfg.get("encoder_name", "bert-base-uncased"),
        pretrained=bool(cfg.get("pretrained", True)),
        hidden_size=int(cfg.get("hidden_size", 256)),
        lstm_hidden=int(cfg.get("lstm_hidden", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_length=max_len,
    )
    model = CLISR(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    steps = max(1, len(train_loader) * int(cfg["epochs"]))
    warmup = int(steps * float(cfg.get("warmup_ratio", 0.1)))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = (step - warmup) / float(max(1, steps - warmup))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    out_dir = Path(cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    bad = 0
    history = []

    for epoch in range(int(cfg["epochs"])):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["view_ids"],
                batch["view_mask"],
            )
            cl = infonce_loss(out["h"], out["h2"], float(cfg["temperature"]))
            isr = F.cross_entropy(out["stance_logits"], batch["stance"])
            cls = F.cross_entropy(out["logits"], batch["labels"])
            reg = l2_regularization((p for p in model.parameters() if p.requires_grad), float(cfg["l2_lambda"]))
            loss = total_loss(
                cl, isr, cls, reg, float(cfg["alpha_cl"]), float(cfg["alpha_isr"]), float(cfg["alpha_cls"])
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": running / max(len(train_loader), 1), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(row)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            bad = 0
            torch.save({"model": model.state_dict(), "config": model_cfg.__dict__}, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= int(cfg.get("patience", 5)):
                print("early stopping")
                break

    try:
        ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"history": history, "test": test_metrics}, f, indent=2)
    print("test", test_metrics)
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.no_pretrained:
        cfg["pretrained"] = False
    if args.pretrained:
        cfg["pretrained"] = True
    train(cfg)


if __name__ == "__main__":
    main()
