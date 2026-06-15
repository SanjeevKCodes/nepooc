# NepOOC-M: Bilingual Nepali-English Benchmark and Comparative Analysis of Multimodal Architectures for OOC Detection

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/theonlysanjeev/nepal-ooc-misinformation)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**Sanjeev Khatiwada — Kathmandu, Nepal — skhatiwada558@gmail.com**

---

## Overview

NepOOC is the first publicly available Nepali-dominant, bilingual benchmark for **out-of-context (OOC) multimodal misinformation detection**. OOC misinformation pairs an authentic, unmanipulated image with a misleading caption — making detection a problem of image-caption semantic alignment rather than image forensics.

| Property | Value |
|----------|-------|
| Total samples | 1,090 image-caption pairs |
| Balance | 545 Pristine / 545 OOC |
| Splits | Train 754 / Val 108 / Test 228 |
| Languages | Nepali 78.5%, English 14.5%, Code-switched 7.0% |
| Typologies | 5 (Fabricated, Miscaptioned, Temporal, Geographic, Identity) |
| IAA (typology) | Cohen's κ = 0.84 |
| IAA (binary) | Cohen's κ = 0.81 |

---

## Key Results

| Model | Type | Macro-F1 (%) | AUC |
|-------|------|:---:|:---:|
| **ResNet-50+mBERT** | Multimodal | **94.65 ± 0.20** | 0.9662 ± 0.0142 |
| mBERT | Text-only | **94.65 ± 0.20** | 0.9697 ± 0.0126 |
| ViT+MuRIL | Multimodal | 93.33 ± 0.37 | 0.9505 ± 0.0057 |
| ViT+TCN | Multimodal | 92.10 ± 1.36 | 0.9616 ± 0.0064 |
| CNN+LSTM | Multimodal | 78.15 ± 10.97 | 0.8548 ± 0.1203 |
| CLIP | Multimodal | 69.00 ± 0.72 | 0.7127 ± 0.0028 |

> **Key finding:** Text-only mBERT is statistically equivalent to the best multimodal system (McNemar median p = 1.000, 0/5 seeds significant at α = 0.05).

---

## Repository Structure
---

## Dataset

Full dataset hosted on Hugging Face:
**https://huggingface.co/datasets/theonlysanjeev/nepal-ooc-misinformation**

Fields: `post_id`, `image_url`, `caption`, `label` (0=Pristine, 1=OOC), `misinformation_type`, `named_entities`, `categories`, `posted_date`, `language`, `article_url`, `article_title`

---

## How to Reproduce

### Step 1 — Clone the repo
```bash
git clone https://github.com/SanjeevKCodes/nepooc.git
cd nepooc
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run all experiments (recommended)
```bash
python src/Final_unified.py
```
Runs all 7 models × 5 seeds × 4 data fractions. Produces Tables VI–XV from the paper.

### Step 4 — Or run individual models
```bash
python src/Resnet_mBert.py    # Best model: ResNet-50 + mBERT
python src/Vit_Muril.py       # ViT-B/16 + MuRIL + LoRA
python src/Vit_Tcn.py         # ViT-B/16 + TCN
python src/CNN_LSTMs.py       # CNN + BiLSTM
python src/clip.py            # CLIP ViT-B/32
```

> **Note:** A GPU is required. Scripts were developed and tested on Kaggle with NVIDIA T4 GPU (free tier). Total compute: ~120 GPU-hours.

---

## Training Configuration

| Model | Optimizer | LR | Batch | Epochs | Scheduler |
|-------|-----------|-----|-------|--------|-----------|
| CNN+LSTM | Adam | 1e-4 | 32 | 80 | StepLR (step=10) |
| ViT+TCN | AdamW | 5e-5 | 32 | 100 | Cosine+WU |
| ResNet-50+mBERT | AdamW | vis:1e-4, txt:2e-5 | 32 | 50 | Cosine+WU |
| CLIP | AdamW | adaptive | 8 | 100 | — |
| ViT+MuRIL | AdamW | adaptive | 8 | 100 | Cosine+WU |

All models trained on 5 seeds: {42, 123, 456, 789, 2024}
Early stopping patience: 10–12 epochs on validation Macro-F1

---

## Citation

```bibtex
@misc{khatiwada2026nepooc,
  author = {Khatiwada, Sanjeev},
  title  = {NepOOC-M: Bilingual Nepali-English Benchmark and Comparative
            Analysis of Multimodal Architectures for OOC Detection},
  year   = {2026},
  url    = {https://github.com/SanjeevKCodes/nepooc},
  note   = {Dataset: https://huggingface.co/datasets/theonlysanjeev/nepal-ooc-misinformation}
}
```

---

## License

This project is licensed under **CC BY-NC 4.0** (research-only, non-commercial).
See [LICENSE](LICENSE) for details.

---

## Contact

Sanjeev Khatiwada | skhatiwada558@gmail.com | Kathmandu, Nepal
