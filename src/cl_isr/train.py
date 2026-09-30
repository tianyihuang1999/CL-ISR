"""Joint training of CL-ISR (Eq. 16) with warmup + cosine schedule and early stopping."""

from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MisleadingTextDataset, SimpleTokenizer, collate
from .losses import infonce_loss, l2_regularization, total_loss, stance_loss
from .metrics import compute_metrics
from .model import build_model, CLISRConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tokenizer(cfg: dict):
    if not cfg.get("pretrained", True):
        return SimpleTokenizer(vocab_size=int(cfg.get("vocab_size", 30522)), max_length=int(cfg["max_length"]))
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(cfg["encoder_name"])


def load_checkpoint(path, device="cpu"):
    # Only tensors and primitive metadata are stored. No pickle objects are needed.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Legacy checkpoint: retrain with the current code; old hash-tokenizer IDs cannot be recovered reliably")
    model = build_model(CLISRConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint


def checkpoint_tokenizer(path, checkpoint):
    config = checkpoint["config"]
    if not config["pretrained"]:
        return build_tokenizer(config)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(Path(path).parent / "tokenizer", local_files_only=True)


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
    cfg = dict(cfg)
    if int(cfg["epochs"]) < 1 or int(cfg["batch_size"]) < 1 or int(cfg["max_length"]) < 1:
        raise ValueError("epochs, batch_size, and max_length must be positive")
    if int(cfg.get("patience", 5)) < 1 or float(cfg.get("min_delta", 0.0)) < 0:
        raise ValueError("patience must be positive and min_delta non-negative")
    for name in ("alpha_cl", "alpha_isr", "alpha_cls", "l2_lambda"):
        if not np.isfinite(float(cfg[name])) or float(cfg[name]) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    is_baseline = cfg.get("model_kind", "cl_isr") == "transformer"
    use_cl = not is_baseline and bool(cfg.get("use_cl", True)) and float(cfg["alpha_cl"]) > 0
    use_isr = not is_baseline and bool(cfg.get("use_isr", True))
    if use_cl and int(cfg["batch_size"]) < 2:
        raise ValueError("Contrastive learning requires batch_size >= 2")
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = int(cfg["max_length"])
    tokenizer = build_tokenizer(cfg)
    train_ds = MisleadingTextDataset(
        cfg["train_path"], tokenizer, max_len, cfg.get("augmentation", "hybrid"), train=True, use_cl=use_cl
    )
    val_ds = MisleadingTextDataset(cfg["val_path"], tokenizer, max_len, train=False)
    test_ds = MisleadingTextDataset(cfg["test_path"], tokenizer, max_len, train=False)
    if use_cl and len(train_ds) < 2:
        raise ValueError("Contrastive learning requires at least two training examples")
    if use_isr and float(cfg["alpha_isr"]) > 0 and train_ds.df["stance"].isna().any():
        warnings.warn("Missing stance labels are excluded from ISR supervision; this is not a fully stance-supervised reproduction")
    loader_options = {"num_workers": int(cfg.get("num_workers", 0)), "worker_init_fn": seed_worker}
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=collate,
        generator=generator, **loader_options
    )
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate, **loader_options)
    test_loader = DataLoader(test_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate, **loader_options)

    model_cfg = CLISRConfig(
        vocab_size=len(tokenizer) if cfg.get("pretrained", True) else tokenizer.vocab_size,
        encoder_name=cfg.get("encoder_name", "bert-base-uncased"),
        pretrained=bool(cfg.get("pretrained", True)),
        hidden_size=int(cfg.get("hidden_size", 256)),
        lstm_hidden=int(cfg.get("lstm_hidden", 128)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_length=max_len,
        use_cl=use_cl,
        use_isr=use_isr,
        fusion=cfg.get("fusion", "gated"),
        freeze_isr_backbone=bool(cfg.get("freeze_isr_backbone", False)),
        model_kind=cfg.get("model_kind", "cl_isr"),
    )
    model = build_model(model_cfg).to(device)
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
    if model_cfg.pretrained:
        tokenizer.save_pretrained(out_dir / "tokenizer")
    best_val_loss = float("inf")
    min_delta = float(cfg.get("min_delta", 0.0))
    best_epoch = None
    bad = 0
    history = []

    for epoch in range(int(cfg["epochs"])):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["input_ids"], batch["attention_mask"])
            cls = F.cross_entropy(out["logits"], batch["labels"])
            cl = cls.new_zeros(())
            if use_cl:
                h1, h2 = model.encode_views(batch["view_a_ids"], batch["view_a_mask"], batch["view_b_ids"], batch["view_b_mask"])
                cl = infonce_loss(h1, h2, float(cfg["temperature"]))
            isr = stance_loss(out["stance_logits"], batch["stance"]) if use_isr else cls.new_zeros(())
            reg = l2_regularization((p for p in model.parameters() if p.requires_grad), float(cfg["l2_lambda"]))
            loss = total_loss(
                cl, isr, cls, reg, float(cfg["alpha_cl"]), float(cfg["alpha_isr"]), float(cfg["alpha_cls"])
            )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; check data and hyperparameters")
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
        if not np.isfinite(val_metrics["loss"]):
            raise ValueError("Non-finite validation loss")
        if val_metrics["loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            bad = 0
            torch.save({"format_version": 2, "model": model.state_dict(), "config": asdict(model_cfg),
                        "training_config": cfg, "epoch": best_epoch, "val_metrics": val_metrics,
                        "tokenizer_kind": "huggingface" if model_cfg.pretrained else "sha256-v1"}, out_dir / "best.pt")
        else:
            bad += 1
            if bad >= int(cfg.get("patience", 5)):
                print("early stopping")
                break

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "best_epoch": best_epoch, "history": history, "test": test_metrics}, f, indent=2)
    print("test", test_metrics)
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pretrained", action="store_true")
    group.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, help="Independent runs, e.g. 42 43 44")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.no_pretrained:
        cfg["pretrained"] = False
    if args.pretrained:
        cfg["pretrained"] = True
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.seeds:
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds must contain distinct values")
        root = Path(cfg.get("output_dir", "outputs"))
        results = [train({**cfg, "seed": seed, "output_dir": str(root / f"seed-{seed}")}) for seed in args.seeds]
        summary = {key: {"mean": float(np.mean([r[key] for r in results])),
                         "std": float(np.std([r[key] for r in results]))} for key in results[0]}
        with open(root / "summary.json", "w", encoding="utf-8") as f:
            json.dump({"seeds": args.seeds, "runs": results, "summary": summary}, f, indent=2)
        print(summary)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
