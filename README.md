# Multimodal Visual Question Answering on VizWiz

Empirical study of multimodal VQA architectures on the [VizWiz dataset](https://vizwiz.org/) — a benchmark of real images taken by blind users paired with spoken questions. The dataset is notably challenging due to low image quality, motion blur, and highly variable answer distributions.

**Course:** CSCI 5922 — Neural Networks | University of Colorado Boulder

---

## Results

| Model | Architecture | VQA Accuracy |
|-------|-------------|-------------|
| MultiModalTransformer | Custom cross-attention (end-to-end) | **0.5352** |
| CLIPGenerator | MLP on frozen CLIP embeddings | 0.5346 |
| CLIPGeneratorV2 | Cross-attention + cosine similarity on CLIP | 0.5050 |

---

## Approaches

### Challenges 1 & 2 — MultiModalTransformer (end-to-end)

A custom multimodal architecture trained directly on raw images and tokenized questions.

- **Image encoder:** patch-based Conv2d projection (32×32 patches → 49 tokens) with learned positional embeddings
- **Text encoder:** word-level embedding with learned positional encoding
- **Fusion:** separate self-attention over image and text sequences, followed by cross-attention where text tokens attend to image patch keys/values
- **Challenge 1:** binary answerability classification head
- **Challenge 2:** closed-set answer generation head over top-3000 answer vocabulary, trained with frequency-weighted cross-entropy loss and augmented images (random crop, horizontal flip, color jitter)

### Challenges 3 & 4 — CLIP-based Models

Operating on frozen 512-dim CLIP embeddings provided with the dataset rather than raw pixels.

- **CLIPGenerator:** projects image and text CLIP features independently, concatenates, and feeds through an MLP classifier
- **CLIPGeneratorV2:** adds a cross-attention layer between projected features and appends cosine similarity between raw CLIP embeddings as an additional input feature

---

## Setup

This notebook is designed to run on **Google Colab** with a GPU runtime (A100 recommended).

### Data

The following are downloaded automatically in the notebook:
- VizWiz annotations (`train.json`, `val.json`, `test.json`)
- VizWiz images (`train.zip`, `val.zip`, `test.zip`)

The CLIP feature file (`VizWiz_CLIP.zip`) must be downloaded manually from your course Canvas and placed at `/content/vizwiz/VizWiz_CLIP.zip` before running Challenges 3 and 4.

### Training details

| Setting | Value |
|---------|-------|
| Training subset | 10,000 samples (50% of train set), `random.seed(42)` |
| Optimizer | AdamW |
| Scheduler | Cosine annealing |
| Early stopping | Patience = 5 |
| Batch size | 64 |
| Hardware | A100 GPU (Google Colab) |

---

## Repo Structure

```
VizWiz-Multimodal-VQA/
├── VizWiz_VQA_Multimodal.ipynb   # Main notebook (all 4 challenges)
└── README.md
```

---

## Key Observations

- The MultiModalTransformer matches CLIPGenerator on VQA accuracy (0.5352 vs 0.5346) despite training on raw pixels rather than pretrained features, suggesting the cross-attention fusion captures relevant visual-linguistic alignment.
- CLIPGeneratorV2's cosine similarity feature did not improve over the base CLIPGenerator, likely because CLIP features are already well-aligned in the same embedding space, making explicit similarity redundant.
- A significant portion of ground truth answers fall outside the top-3000 vocabulary, putting a ceiling on closed-set generation accuracy regardless of model quality.
