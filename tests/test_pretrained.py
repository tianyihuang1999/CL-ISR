from dataclasses import asdict

import pytest
import torch
from transformers import BertConfig, BertModel, RobertaConfig

from cl_isr.model import CLISRConfig, build_model
from cl_isr.train import load_checkpoint


def tiny_bert():
    return BertConfig(vocab_size=32, hidden_size=12, num_hidden_layers=1,
                      num_attention_heads=3, intermediate_size=24, max_position_embeddings=32)


@pytest.mark.parametrize("frozen", [False, True])
def test_pretrained_isr_gradient_and_freeze_behavior(tmp_path, frozen):
    path = tmp_path / "bert"
    BertModel(tiny_bert()).save_pretrained(path)
    model = build_model(CLISRConfig(encoder_name=str(path), hidden_size=8, lstm_hidden=4,
                                    freeze_isr_backbone=frozen))
    model.train()
    assert model.isr.embedder.training is (not frozen)
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    out = model(ids, ids.ne(0).long())
    (out["logits"].sum() + out["stance_logits"].sum()).backward()
    grads = [p.grad for p in model.isr.embedder.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads) is (not frozen)
    # Reload without accessing the original pretrained directory or the network.
    ckpt = tmp_path / "best.pt"
    config = asdict(model.config)
    config["encoder_name"] = "unavailable-pretrained-source"
    torch.save({"format_version": 2, "model": model.state_dict(), "config": config}, ckpt)
    loaded, _ = load_checkpoint(ckpt)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(ids, ids.ne(0).long())["logits"], loaded(ids, ids.ne(0).long())["logits"])


@pytest.mark.parametrize("kind", ["bert", "roberta"])
def test_standalone_transformer_baselines(kind):
    config = tiny_bert() if kind == "bert" else RobertaConfig(
        vocab_size=32, hidden_size=12, num_hidden_layers=1,
        num_attention_heads=3, intermediate_size=24, max_position_embeddings=32)
    model = build_model(CLISRConfig(model_kind="transformer", backbone_config=config.to_dict()))
    ids = torch.tensor([[2, 3, 4], [2, 5, 6]])
    logits = model(ids, torch.ones_like(ids))["logits"]
    assert logits.shape == (2, 2)
    assert not hasattr(model, "isr")
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()


def test_pretrained_training_saves_portable_tokenizer(tmp_path):
    from pathlib import Path
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast
    from torch.utils.data import DataLoader
    from cl_isr.data import MisleadingTextDataset, collate
    from cl_isr.train import train, load_config, checkpoint_tokenizer, evaluate

    source = tmp_path / "source"
    BertModel(tiny_bert()).save_pretrained(source)
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "news": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")
    fast.save_pretrained(source)
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/smoke.yaml")
    cfg.update(pretrained=True, encoder_name=str(source), hidden_size=8, lstm_hidden=4,
               max_length=12, epochs=1, output_dir=str(tmp_path / "trained"))
    for split in ["train", "val", "test"]:
        cfg[f"{split}_path"] = str(root / f"data/sample/{split}.csv")
    expected = train(cfg)
    path = tmp_path / "trained" / "best.pt"
    # The source model directory can disappear; evaluation uses saved artifacts.
    source.rename(tmp_path / "source-moved")
    model, checkpoint = load_checkpoint(path)
    saved = checkpoint_tokenizer(path, checkpoint)
    ds = MisleadingTextDataset(cfg["test_path"], saved, 12, train=False)
    actual = evaluate(model, DataLoader(ds, batch_size=4, collate_fn=collate), "cpu")
    assert actual == expected
