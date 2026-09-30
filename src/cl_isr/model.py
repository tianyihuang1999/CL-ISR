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


class ContrastiveEncoder(nn.Module):
    def __init__(self, config: CLISRConfig):
        super().__init__()
        self.pretrained = config.pretrained
        if config.pretrained:
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained(config.encoder_name)
            bert_dim = self.backbone.config.hidden_size
        else:
            self.backbone = nn.Embedding(config.vocab_size, config.hidden_size)
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
        if config.pretrained:
            from transformers import AutoModel

            self.embedder = AutoModel.from_pretrained(config.encoder_name)
            in_dim = self.embedder.config.hidden_size
            for p in self.embedder.parameters():
                p.requires_grad = False
        else:
            self.embedder = nn.Embedding(config.vocab_size, config.hidden_size)
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
        self.stance_head = nn.Linear(att_dim, config.num_stance)
        self.out_proj = nn.Linear(att_dim, config.hidden_size)

    def _token_emb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            with torch.no_grad():
                return self.embedder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.embedder(input_ids)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
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
        stance_logits = self.stance_head(s_raw)
        s = self.out_proj(s_raw)
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
        self.cl_encoder = ContrastiveEncoder(config)
        self.isr = ImplicitStanceReasoning(config)
        self.fusion = FusionLayer(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def encode_views(self, ids_a, mask_a, ids_b, mask_b):
        h1 = self.cl_encoder(ids_a, mask_a)
        h2 = self.cl_encoder(ids_b, mask_b)
        return h1, h2

    def forward(self, input_ids, attention_mask, view_ids=None, view_mask=None):
        h = self.cl_encoder(input_ids, attention_mask)
        s, stance_logits, alpha = self.isr(input_ids, attention_mask)
        r, beta = self.fusion(h, s)
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
