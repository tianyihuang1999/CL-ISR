"""CL-ISR architecture (paper Section III).

- Contrastive encoder f_theta maps text to h (Eq. 1)
- ISR encoder l_phi is BiLSTM + token attention producing stance vector s (Eq. 5-9)
- Fusion uses a sigmoid gate over (h, s) (Eq. 13-14)
- Classification head predicts real vs misleading (Eq. 15)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CLISRConfig:
    vocab_size: int = 30522
    encoder_name: str = "bert-base-uncased"
    pretrained: bool = True
    hidden_size: int = 256
    lstm_hidden: int = 128
    num_labels: int = 2
    num_stance: int = 3
    dropout: float = 0.1
    max_length: int = 512
    use_cl: bool = True
    use_isr: bool = True
    fusion: str = "gated"
    freeze_isr_backbone: bool = False
    model_kind: str = "cl_isr"
    backbone_config: dict | None = None


def build_backbone(config: CLISRConfig, sequence_classifier: bool = False):
    """Use saved architecture at evaluation, without downloading pretrained weights."""
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

    factory = AutoModelForSequenceClassification if sequence_classifier else AutoModel
    if config.backbone_config is not None:
        values = dict(config.backbone_config)
        model_type = values.pop("model_type")
        return factory.from_config(AutoConfig.for_model(model_type, **values))
    kwargs = {"num_labels": config.num_labels} if sequence_classifier else {}
    return factory.from_pretrained(config.encoder_name, **kwargs)


class ContrastiveEncoder(nn.Module):
    def __init__(self, config: CLISRConfig):
        super().__init__()
        self.pretrained = config.pretrained
        if config.pretrained:
            self.backbone = build_backbone(config)
            bert_dim = self.backbone.config.hidden_size
        else:
            self.backbone = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
            bert_dim = config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(bert_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0]
        else:
            emb = self.backbone(input_ids)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.proj(pooled)


class ImplicitStanceReasoning(nn.Module):
    """BiLSTM + additive attention stance encoder (Eq. 7-9)."""

    def __init__(self, config: CLISRConfig):
        super().__init__()
        self.pretrained = config.pretrained
        self.freeze_backbone = config.freeze_isr_backbone
        if config.pretrained:
            self.embedder = build_backbone(config)
            in_dim = self.embedder.config.hidden_size
            if self.freeze_backbone:
                self.embedder.requires_grad_(False)
                self.embedder.eval()
        else:
            self.embedder = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
            in_dim = config.hidden_size
        self.lstm = nn.LSTM(
            in_dim,
            config.lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        att_dim = config.lstm_hidden * 2
        self.W_h = nn.Linear(att_dim, att_dim)
        self.v = nn.Linear(att_dim, 1, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.stance_head = nn.Linear(config.hidden_size, config.num_stance)
        self.out_proj = nn.Identity() if att_dim == config.hidden_size else nn.Linear(att_dim, config.hidden_size)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.pretrained and self.freeze_backbone:
            self.embedder.eval()
        return self

    def _token_emb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            if self.freeze_backbone:
                with torch.no_grad():
                    return self.embedder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return self.embedder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.embedder(input_ids)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if (attention_mask.sum(1) == 0).any():
            raise ValueError("ISR requires at least one unmasked token per sample")
        x = self._token_emb(input_ids, attention_mask)
        lengths = attention_mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        H, _ = self.lstm(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(H, batch_first=True, total_length=input_ids.size(1))
        scores = self.v(torch.tanh(self.W_h(H))).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        alpha = torch.softmax(scores, dim=-1)
        s_raw = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)
        s_raw = self.dropout(s_raw)
        s = self.out_proj(s_raw)
        stance_logits = self.stance_head(s)
        return s, stance_logits, alpha


class FusionLayer(nn.Module):
    """Dynamic fusion r = beta * h + (1 - beta) * s (Eq. 13-14)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, 1, bias=False)
        self.W_s = nn.Linear(hidden_size, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta = torch.sigmoid(self.W_h(h) + self.W_s(s) + self.b)
        r = beta * h + (1.0 - beta) * s
        return r, beta


class CLISR(nn.Module):
    def __init__(self, config: CLISRConfig):
        super().__init__()
        self.config = config
        if config.fusion not in {"gated", "mean"}:
            raise ValueError("fusion must be 'gated' or 'mean'")
        self.cl_encoder = ContrastiveEncoder(config)
        self.isr = ImplicitStanceReasoning(config) if config.use_isr else None
        self.fusion = FusionLayer(config.hidden_size) if config.use_isr and config.fusion == "gated" else None
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        if config.pretrained:
            config.backbone_config = self.cl_encoder.backbone.config.to_dict()

    def encode_views(self, ids_a, mask_a, ids_b, mask_b):
        h1 = self.cl_encoder(ids_a, mask_a)
        h2 = self.cl_encoder(ids_b, mask_b)
        return h1, h2

    def forward(self, input_ids, attention_mask, view_ids=None, view_mask=None):
        h = self.cl_encoder(input_ids, attention_mask)
        s = stance_logits = alpha = beta = None
        r = h
        if self.isr is not None:
            s, stance_logits, alpha = self.isr(input_ids, attention_mask)
            if self.fusion is not None:
                r, beta = self.fusion(h, s)
            else:
                beta = h.new_full((h.size(0), 1), 0.5)
                r = (h + s) * 0.5
        logits = self.classifier(r)
        h2 = None
        if view_ids is not None:
            h2 = self.cl_encoder(view_ids, view_mask)
        return {
            "logits": logits,
            "h": h,
            "h2": h2,
            "s": s,
            "stance_logits": stance_logits,
            "beta": beta,
            "alpha": alpha,
            "r": r,
        }


class TransformerClassifier(nn.Module):
    """A standalone BERT/RoBERTa classification baseline without CL/ISR/fusion."""

    def __init__(self, config: CLISRConfig):
        super().__init__()
        if not config.pretrained:
            raise ValueError("Transformer baselines require pretrained: true")
        self.config = config
        self.backbone = build_backbone(config, sequence_classifier=True)
        config.backbone_config = self.backbone.config.to_dict()

    def forward(self, input_ids, attention_mask):
        return {"logits": self.backbone(input_ids=input_ids, attention_mask=attention_mask).logits}


def build_model(config: CLISRConfig):
    if config.model_kind == "cl_isr":
        return CLISR(config)
    if config.model_kind == "transformer":
        return TransformerClassifier(config)
    raise ValueError(f"Unknown model_kind: {config.model_kind}")
