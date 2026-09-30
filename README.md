# CL-ISR

PyTorch implementation of **CL-ISR: A Contrastive Learning and Implicit Stance
Reasoning Framework for Misleading Text Detection on Social Media**, IEEE ICECAI
2025. DOI: [10.1109/ICECAI66283.2025.11170636](https://doi.org/10.1109/ICECAI66283.2025.11170636).

The implementation follows the paper's contrastive objective, BiLSTM attention
stance encoder, gated fusion, and joint optimization. The bundled CSVs contain
only 30 training, 6 validation, and 6 test examples for pipeline checks. They do
not reproduce the paper's experimental results. See [implementation notes](docs/reproduction.md)
for equation mappings, assumptions, and unresolved experimental details.

## Install

Python 3.9+; the paper used Python 3.9, PyTorch 1.13.1, and Transformers 4.21.0.
The dependency ranges also permit newer releases; exact numerical reproduction
requires recording the environment used for each experiment.

```sh
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
```

## Train and evaluate

CPU smoke run, with a stable SHA-256 tokenizer and no model downloads:

```sh
python scripts/train.py --config configs/smoke.yaml --no-pretrained
python scripts/evaluate.py --config configs/smoke.yaml --ckpt outputs/smoke/best.pt
```

Full architecture with pretrained BERT (downloads weights on the first run):

```sh
python scripts/train.py --config configs/default.yaml
```

Replace the sample CSV paths with your prepared data before running experiments.
The default configuration uses batch size 16, sequence length 512, AdamW,
warmup/cosine scheduling, and up to 50 epochs. Checkpoint selection and early
stopping use **validation classification loss**, with patience 5. `min_delta`
(default 0) controls the required decrease. `max_length` is honored in both
pretrained and smoke modes. `num_workers` controls data loading.

Each output directory contains `best.pt` and `history.json`, plus `tokenizer/`
for pretrained models. Checkpoints include architecture, training configuration,
selected epoch, validation metrics, and the tokenizer scheme. Evaluation loads
architecture, tokenizer, and sequence length from the checkpoint; YAML supplies
only the data path and evaluation batch size. Keep `tokenizer/` beside `best.pt`.
Evaluation reconstructs transformer architecture locally without downloading
original pretrained weights.

Checkpoints created before format version 2 must be retrained. The old smoke
tokenizer used process-randomized hashes, and the updated stance representation
also changes model state. Loading those files silently would produce invalid
comparisons.

Three independent runs (Section IV-B), with separate checkpoints and mean /
population standard deviation in `summary.json`:

```sh
python scripts/train.py --config configs/default.yaml --seeds 42 43 44
```

Cross-domain evaluation uses a new CSV with the trained tokenizer:

```sh
python scripts/evaluate.py --ckpt outputs/seed-42/best.pt --data data/target/test.csv
```

Metrics are accuracy, binary recall and binary F1 for label 1 (misleading).

## Model and supervision

- Original text feeds the classification and stance branches.
- Two independently augmented views feed the contrastive encoder. The matching
  view is the positive; other samples in the batch are negatives.
- A transformer CLS representation (or masked embedding mean in smoke mode)
  passes through an MLP to produce `h`.
- A separate token encoder feeds BiLSTM and additive attention to produce `s`.
  The stance head and fusion use this same feature. If dimensions differ, an
  explicit projection aligns them; otherwise the weighted BiLSTM state is used.
- A learned sigmoid gate computes `r = beta * h + (1 - beta) * s`, followed by
  binary classification.
- Training combines InfoNCE, annotated-stance cross entropy, classification
  cross entropy, and one explicit L2 term over trainable parameters. AdamW weight
  decay is also retained, as specified in the experimental setup.

Both transformer branches are trainable by default. `freeze_isr_backbone: true`
provides an explicit memory-saving variant; its frozen backbone stays in eval
mode, including during training. The paper does not specify this freeze option.

## Data

CSV columns:

| Column | Values |
| --- | --- |
| `text` | Non-empty post or headline |
| `label` | 0 = real, 1 = misleading |
| `stance` | Optional: -1 = opposition, 0 = neutral, 1 = support |

Missing stance cells or a missing column mean **unannotated**, and are excluded
from stance loss. A batch without stance annotations contributes zero stance
loss. This is different from a genuine neutral label. Full stance-supervised
reproduction requires actual annotations; no automatic mapping from veracity to
stance is assumed. Invalid labels and empty text are rejected by the dataset.

```sh
python scripts/prepare_data.py --src raw.csv --dest data/fakenewsnet \
  --text-col text --label-col label --stance-col stance --seed 42
```

The preparation script removes missing/empty text and missing labels before type
conversion, preserves unknown stances, rejects conflicting duplicate labels,
deduplicates exact text before splitting, and produces stratified 80/10/10 splits.
Very small class counts may be insufficient for stratification. Event/thread
splits require a dataset-specific preparation pipeline; exact-text deduplication
does not prevent event-level leakage.

Optional `--clean-text` removes URLs and Unicode punctuation/symbols (including
emoji), and normalizes whitespace. `--stopwords-file words.txt` supplies an
explicit UTF-8 list (one word per line) and requires `--clean-text`. This is a
transparent approximation: the paper does not provide its cleaning rules or
stopword lists. Stopword removal is whitespace-token based. Preserve and document
your preprocessing choices consistently for train/validation/test.

Data sources discussed in the paper include
[FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet), PHEME, and Weibo-Misinfo.
The exact sampled posts, annotations, and split files are not bundled here.
For Chinese, choose a suitable encoder such as `bert-base-chinese`, and provide
language-appropriate augmentation. The built-in augmentation is a small English
fallback lexicon and whitespace tokenization; changing the encoder alone does
not make it a Chinese reproduction.

## Baselines and ablations

```sh
python scripts/run_baseline.py --train data/sample/train.csv --test data/sample/test.csv
python scripts/train.py --config configs/bert.yaml
python scripts/train.py --config configs/roberta.yaml
```

BERT and RoBERTa use standalone `AutoModelForSequenceClassification` models with
no ISR or fusion branch. They share the training/evaluation procedure.

For CL-ISR, edit the YAML as follows:

| Experiment | Configuration | Effect |
| --- | --- | --- |
| No contrastive learning | `use_cl: false` | No view generation or contrastive objective; text encoder remains for supervised classification |
| No stance reasoning | `use_isr: false` | Removes the ISR and fusion modules; classify `h` directly |
| No dynamic fusion | `fusion: mean` | Equal-weight fusion with no learned gate |
| No stance supervision only | `alpha_isr: 0` | Retains the ISR features and classification gradients |

Setting `alpha_cl: 0` also disables contrastive view generation. Setting
`alpha_isr: 0` is **not** an ISR removal. A singleton final batch has no negative
pairs and therefore contributes zero InfoNCE while retaining supervised losses.

## License

Code is MIT licensed. The IEEE paper PDF is not redistributed.
