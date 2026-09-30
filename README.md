CL-ISR
Reproduction of Huang et al., ICECAI 2025: contrastive learning plus implicit stance reasoning for misleading social-media text detection.

This repository implements the architecture and training objective described in the paper. Official code was not released; dataset construction for FakeNewsNet, PHEME, and Weibo-Misinfo must be done from the original public sources.

Paper
CL-ISR: A Contrastive Learning and Implicit Stance Reasoning Framework for Misleading Text Detection on Social Media
IEEE ICECAI 2025, DOI: 10.1109/ICECAI66283.2025.11170636

What is implemented
Contrastive encoder f_theta (Eq. 1) with BERT or a lightweight embedding encoder
Positive/negative views via random deletion, synonym replacement, insertion, or hybrid augmentation (Section III-A, Table 3)
InfoNCE loss with temperature 0.07 (Eq. 2)
ISR module: BiLSTM + additive attention + 3-way stance head (Eq. 5-11)
Gated fusion r = beta * h + (1-beta) * s (Eq. 12-14)
Joint loss alpha1 L_CL + alpha2 L_ISR + alpha3 L_cls + L2 (Eq. 16-17)
AdamW, linear warmup, cosine annealing, early stopping (Section IV-B)
SVM (TF-IDF) baseline (Table 1); BERT/RoBERTa baselines are this model with modules ablated by loss weights

Install
Python 3.9+ recommended (paper used 3.9, PyTorch 1.13.1, Transformers 4.21.0).

    pip install -r requirements.txt

Smoke test (no BERT download)

    python scripts/train.py --config configs/smoke.yaml --no-pretrained

Full BERT training on the bundled toy CSV

    python scripts/train.py --config configs/default.yaml --pretrained

SVM baseline

    python scripts/run_baseline.py --train data/sample/train.csv --test data/sample/test.csv

Data format
CSV columns:

text: post or headline
label: 0 = real, 1 = misleading
stance: -1 opposition, 0 neutral, +1 support (optional; defaults to 0)

Convert a public dump:

    python scripts/prepare_data.py --src raw.csv --dest data/fakenewsnet --text-col text --label-col label --stance-col stance

Public datasets mentioned in the paper
FakeNewsNet: https://github.com/KaiDMML/FakeNewsNet
PHEME: rumor threads with stance labels on replies
Weibo-Misinfo: Chinese Weibo posts; the paper's exact dump is not redistributed here. Use bert-base-chinese in configs/default.yaml for Chinese text.

Hyperparameters (paper Section IV-B)
max length 512, batch 16 or 32, lr 2e-5, weight decay 0.01, tau 0.07, lambda 1e-4, up to 50 epochs, patience 5. The smoke config uses a tiny encoder so the pipeline can run on CPU.

Ablation
Set alpha_cl, alpha_isr to 0 in the YAML to drop CL or ISR. Fusion is always present; a mean-fusion ablation can be added by replacing FusionLayer with 0.5 * h + 0.5 * s.

License
Code in this repository is MIT. The paper is IEEE copyrighted; we do not copy the PDF.
