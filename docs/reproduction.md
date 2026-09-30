# Paper-to-code audit

Reference: Huang et al., CL-ISR, IEEE ICECAI 2025,
DOI 10.1109/ICECAI66283.2025.11170636, Sections III-IV.

## Implemented correspondence

| Paper | Implementation |
| --- | --- |
| Eq. 1 | Transformer CLS + projection / lightweight smoke encoder |
| Eq. 2 | One-direction cross-view in-batch InfoNCE with cosine similarity and temperature 0.07 |
| Eqs. 5-9 | Token features, BiLSTM, masked additive attention, stance representation and three-class head |
| Eqs. 12-14 | Sigmoid gate and weighted fusion |
| Eq. 15 | Binary classification cross entropy |
| Eqs. 16-17 | Weighted objectives plus one explicit L2 term over all trainable modules |
| Section IV-A | CSV inputs and stratified 80/10/10 preparation |
| Section IV-B | Batch 16, max length 512, AdamW 2e-5 / decay 0.01, warmup + cosine, maximum 50 epochs, validation-loss patience 5, optional three-seed aggregation |
| Table 2 | Explicit CL removal, ISR removal, and mean-fusion options |
| Table 3 | Deletion, replacement, insertion, and hybrid fallback augmentations |

## Corrections made in this audit

1. Missing stance annotations no longer become fabricated neutral supervision.
   Missing-only batches have a differentiable zero stance loss.
2. Supervised heads consume original text, while InfoNCE consumes two augmented
   views. Previously one augmented view was used for all supervised targets.
3. The ISR transformer is no longer silently frozen. Freezing is configurable
   and also disables its dropout. Joint trainability follows Section III-C;
   the exact choice of a separate transformer token embedder is an implementation
   assumption because the paper does not specify ISR input embeddings.
4. Stance classification and fusion now use the same stance vector. Equal-sized
   BiLSTM states require no additional output projection.
5. ISR removal actually removes the module. Mean fusion removes gate parameters.
   Standalone transformer classifiers replace the incorrectly described baseline
   obtained by merely zeroing auxiliary loss weights.
6. Early stopping and checkpoint selection track validation loss rather than F1.
   This implementation chooses classification validation loss; the paper does not
   specify which validation loss component is monitored.
7. Stable SHA-256 token IDs replace Python's randomized hash. Effective sequence
   length, vocabulary size, tokenizer artifacts and architecture are saved with
   the checkpoint and respected on independent evaluation.
8. Missing data is handled before string/integer conversion. Invalid annotations
   fail explicitly and identical texts cannot cross prepared splits.
9. DataLoader worker count and worker random seeds are honored, and independent
   multi-seed runs have separate output directories and aggregate statistics.

## Interpretations and limits

- Eq. 2's printed denominator uses unprimed candidate features, while its
  numerator and prose describe matched augmented views. We retain the standard
  cross-view InfoNCE interpretation: the denominator includes the positive and
  all opposite-view candidates. It is not a symmetric 2N-view SimCLR loss.
- Eqs. 4 and 11 include module regularizers, while Eqs. 16-17 add a global one.
  We apply global explicit L2 once to avoid double-counting it. AdamW decay is a
  separate mechanism retained from Section IV-B. The final classification head
  is included in the trainable fusion/classification module for this purpose.
- The introduction mentions two stages, but Section III-C describes joint
  optimization without a pretraining schedule. This implementation uses joint
  training and does not invent an undocumented stage schedule.
- Encoder/projection widths, dropout, warmup fraction, augmentation strengths,
  backbone sharing/freezing, loss weights, and random seeds are not fully
  specified. YAML defaults are implementation choices, not recovered settings.
- The paper describes fine-tuning BERT-family models without enough detail to
  uniquely determine a backbone or classification head. Standalone Hugging Face
  BERT/RoBERTa classification models are explicit baseline choices.
- Table 2 does not fully define every ablation. Removing CL is interpreted as
  removing the contrastive objective while retaining a supervised text encoder;
  removing dynamic fusion is implemented as an equal-weight mean.
- The exact sampled datasets, stance annotations on all corpora, PHEME-to-binary
  mapping, cleaning rules, Chinese augmentation, and event/thread split policy
  are unavailable in this repository. No veracity-to-stance mapping is inferred.
- Table 1's heading refers to recall, while the surrounding discussion calls
  some of the same values F1 or accuracy. Code reports all three separately;
  the paper's numbers should not be treated as an unambiguous automated target.
- The bundled synthetic examples only validate the software pipeline. They
  cannot validate reported dataset scores, cross-domain performance, or full
  scientific reproducibility.

## Validation

Regression tests cover stable tokenization across processes, checkpoint reload,
missing/invalid annotations, exact-duplicate leakage, original/augmented inputs,
InfoNCE pair alignment, attention masking, padding invariance, structural
ablations, loss-based early stopping, and tiny local BERT/RoBERTa models including
ISR gradients and frozen dropout behavior. They run without downloading weights.
Large pretrained-model convergence and published benchmark scores require the
original data and substantial training resources.

Audit validation: 28 tests passed on Windows with Python 3.12.14, PyTorch
2.14.0+cpu, and Transformers 5.17.0. The smoke CLI also completed three independent
runs (seeds 42, 43, 44). These checks validate implementation behavior, not the
paper's numerical benchmarks or compatibility with every allowed library version.
