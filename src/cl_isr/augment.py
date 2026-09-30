"""Text views used by contrastive learning (paper Section III-A, Table 3).

Positive pairs are two augmented views of the same original text.
The hybrid strategy is random deletion plus synonym replacement.
"""

from __future__ import annotations

import random
import re
from typing import List, Sequence

_WORD = re.compile(r"\S+")

# Lightweight fallback lexicon so synonym replacement works without NLTK data.
_SYNONYMS = {
    "good": ["great", "fine", "positive"],
    "bad": ["poor", "awful", "negative"],
    "says": ["claims", "states", "reports"],
    "people": ["public", "citizens", "users"],
    "news": ["report", "story", "coverage"],
    "true": ["real", "accurate", "factual"],
    "false": ["fake", "untrue", "misleading"],
    "official": ["authority", "government", "agency"],
    "breaking": ["urgent", "latest", "new"],
    "study": ["research", "paper", "analysis"],
}


def tokenize(text: str) -> List[str]:
    return _WORD.findall(text) or [text]


def detokenize(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def random_deletion(text: str, p: float = 0.15, rng: random.Random | None = None) -> str:
    rng = rng or random
    tokens = tokenize(text)
    if len(tokens) <= 1:
        return text
    kept = [t for t in tokens if rng.random() > p]
    if not kept:
        kept = [rng.choice(tokens)]
    return detokenize(kept)


def synonym_replacement(text: str, n: int = 2, rng: random.Random | None = None) -> str:
    rng = rng or random
    tokens = tokenize(text)
    candidates = [i for i, t in enumerate(tokens) if t.lower() in _SYNONYMS]
    rng.shuffle(candidates)
    for i in candidates[:n]:
        options = _SYNONYMS[tokens[i].lower()]
        tokens[i] = rng.choice(options)
    return detokenize(tokens)


def random_insertion(text: str, n: int = 1, rng: random.Random | None = None) -> str:
    rng = rng or random
    tokens = tokenize(text)
    extras = ["really", "actually", "reportedly", "allegedly", "recently"]
    for _ in range(n):
        tokens.insert(rng.randrange(len(tokens) + 1), rng.choice(extras))
    return detokenize(tokens)


def hybrid_augment(text: str, rng: random.Random | None = None) -> str:
    rng = rng or random
    return synonym_replacement(random_deletion(text, rng=rng), rng=rng)


def augment(text: str, strategy: str = "hybrid", rng: random.Random | None = None) -> str:
    rng = rng or random
    if strategy == "deletion":
        return random_deletion(text, rng=rng)
    if strategy == "synonym":
        return synonym_replacement(text, rng=rng)
    if strategy == "insertion":
        return random_insertion(text, rng=rng)
    if strategy == "hybrid":
        return hybrid_augment(text, rng=rng)
    raise ValueError(f"Unknown augmentation strategy: {strategy}")


def two_views(text: str, strategy: str = "hybrid", rng: random.Random | None = None) -> tuple[str, str]:
    rng = rng or random
    return augment(text, strategy, rng), augment(text, strategy, rng)
