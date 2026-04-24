# Multimodal VQA for Visual Accessibility

Visual Question Answering system built for the [VizWiz dataset](https://vizwiz.org/) — a benchmark of real photos taken by blind users paired with spoken questions. The dataset is uniquely difficult: images are often blurry or poorly framed, questions are conversational, and answers are highly variable across annotators.

The project explores two approaches to multimodal fusion: training directly on raw images with a custom transformer architecture, and leveraging frozen CLIP embeddings for faster, feature-level reasoning.

---

## Results

| Model | VQA Accuracy |
|-------|-------------|
| MultiModalTransformer (raw images + cross-attention) | **0.5352** |
| CLIPGenerator (frozen CLIP embeddings) | 0.5346 |
| CLIPGeneratorV2 (CLIP + cosine similarity feature) | 0.5050 |

---

## Models

### MultiModalTransformer

A custom end-to-end architecture trained on raw images and tokenized questions.

- Images are split into 49 patches (32×32) and projected to a shared embedding space
- Questions are encoded with a word-level vocabulary and learned positional embeddings
- Separate self-attention runs over image patches and text tokens independently
- Cross-attention fuses them: text tokens attend to image patch keys/values
- Two output heads: binary answerability classification and closed-set answer generation

Training used frequency-weighted cross-entropy loss to reduce bias toward the most common answers, along with augmented images (random crop, horizontal flip, color jitter).

### CLIPGenerator

Projects frozen 512-dim CLIP image and text features independently, concatenates them, and passes through an MLP to predict answers. Fast to train and competitive with the end-to-end model.

### CLIPGeneratorV2

Extends CLIPGenerator with a cross-attention layer between projected features and appends the cosine similarity between raw CLIP embeddings as an explicit input. Slightly underperforms the base model — likely because CLIP features are already well-aligned, making the similarity signal redundant.

---

## Setup

Runs on **Google Colab** with a GPU runtime.

The notebook handles downloading VizWiz images and annotations automatically. CLIP feature files need to be sourced separately and placed at `/content/vizwiz/VizWiz_CLIP.zip` before running the CLIP-based models.

### Training configuration

| Setting | Value |
|---------|-------|
| Training subset | 10,000 samples, `random.seed(42)` |
| Optimizer | AdamW |
| LR scheduler | Cosine annealing |
| Early stopping | Patience = 5 |
| Batch size | 64 |
| Hardware | A100 GPU |

---

## Observations

**Answer distribution collapse.** Without loss reweighting, all models converged to predicting high-frequency answers — particularly "unanswerable" and "yes" — within the first few epochs, achieving superficially acceptable top-1 accuracy while essentially ignoring question content. Switching to log-smoothed inverse frequency weighting (`1 / (log1p(count) + 1)`) stabilized training and produced meaningful answer diversity. Raw inverse frequency weighting was too aggressive and caused instability; the log smoothing was necessary to keep gradients well-behaved.

**Vocabulary coverage is the real ceiling.** Even with a 3,000-word answer vocabulary covering the most common training answers, a substantial portion of validation ground truth answers fall entirely outside the vocabulary. No architectural improvement can recover these — the ceiling on closed-set generation accuracy is set by vocabulary design, not model capacity. Expanding to a larger vocabulary or moving to open-ended generation would be the meaningful next step.

**End-to-end vs. CLIP features.** The MultiModalTransformer matches CLIPGenerator in VQA accuracy (0.5352 vs 0.5346) despite having no pretrained visual backbone whatsoever — it learns patch embeddings from scratch on 10,000 training images. This suggests the cross-attention fusion is effectively learning to align image regions with question tokens, rather than relying on rich visual representations. Whether this holds at larger data scales is an open question.

**Added complexity in CLIPGeneratorV2 hurt rather than helped.** Adding cosine similarity between CLIP image and text features as an explicit input, combined with cross-attention between projected features, dropped accuracy from 0.5346 to 0.5050. CLIP image and text encoders are trained to align in the same embedding space, so the cosine similarity is already implicitly captured in the concatenated features — making it explicit adds a redundant signal that likely interferes with learning rather than helping. Simpler fusion outperformed the more complex variant.

**Image augmentation mattered more than architecture depth.** Increasing model depth (more transformer layers, larger d_model) produced diminishing returns on validation accuracy and sometimes hurt due to overfitting on the 10K training subset. In contrast, augmentation — particularly random cropping and color jitter — had a more consistent positive effect, likely because VizWiz images are already visually diverse and noisy, so augmentation better matches the distribution the model encounters at test time.

**The answerability task is deceptively easy.** Binary classification of whether a question is answerable reached high validation accuracy quickly, but this is partly a dataset artifact — the class imbalance in VizWiz means a model predicting "answerable" for most inputs can appear to perform well. The more informative evaluation is VQA accuracy on the generation task, which penalizes both wrong answers and unanswerable predictions.
