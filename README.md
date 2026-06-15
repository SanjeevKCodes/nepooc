NepOOC-M: Bilingual Nepali-English Benchmark and Comparative Analysis of Multimodal Architectures for OOC Detection

Sanjeev Khatiwada — Kathmandu, Nepal — skhatiwada558@gmail.com


Overview

NepOOC is the first publicly available Nepali-dominant, bilingual benchmark for out-of-context (OOC) multimodal misinformation detection. OOC misinformation pairs an authentic, unmanipulated image with a misleading caption — making detection a problem of image-caption semantic alignment rather than image forensics.


1,090 image-caption pairs (545 pristine, 545 OOC)
5 manipulation typologies: Fabricated, Miscaptioned, Temporal Mismatch, Geographic Mismatch, Identity Mismatch
Bilingual: Nepali (78.5%), English (14.5%), code-switched (7.0%)
Inter-annotator agreement: Cohen's κ = 0.84 (typology), κ = 0.81 (binary)
Splits: Train 754 / Validation 108 / Test 228


Key Results

ModelTypeMacro-F1 (%)AUCResNet-50+mBERTMultimodal94.65 ± 0.200.9662 ± 0.0142mBERTText-only94.65 ± 0.200.9697 ± 0.0126ViT+MuRILMultimodal93.33 ± 0.370.9505 ± 0.0057ViT+TCNMultimodal92.10 ± 1.360.9616 ± 0.0064CNN+LSTMMultimodal78.15 ± 10.970.8548 ± 0.1203CLIPMultimodal69.00 ± 0.720.7127 ± 0.0028

Key finding: Text-only mBERT is statistically equivalent to the best multimodal system (McNemar median p = 1.000, 0/5 seeds significant at α = 0.05), indicating caption semantics carry sufficient signal at this dataset scale.

Repository Structure

nepooc/
├── Final_unified.py        # Runs ALL experiments (recommended entry point)
├── Resnet_mBert.py         # ResNet-50 + mBERT (best model)
├── clip.py                 # CLIP ViT-B/32
├── CNN_LSTMs.py            # CNN + BiLSTM
├── Vit_Muril.py            # ViT-B/16 + MuRIL + LoRA
├── Vit_Tcn.py              # ViT-B/16 + TCN
├── train.csv               # 754 samples
├── validation.csv          # 108 samples
├── test.csv                # 228 samples
├── nepOOC_full.csv         # Full 1,090 samples
└── README.md

Dataset

The full dataset is hosted on Hugging Face:
https://huggingface.co/datasets/theonlysanjeev/nepal-ooc-misinformation

Fields: post_id, image_url, caption, label (0=Pristine, 1=OOC), misinformation_type, named_entities, categories, posted_date, language, article_url, article_title

How to Run

Recommended: Run all experiments at once

python# On Kaggle (T4 GPU recommended):
# 1. Upload your dataset files
# 2. Run Final_unified.py
python Final_unified.py

Run individual models

python# ResNet-50 + mBERT (best model)
python Resnet_mBert.py

# CLIP
python clip.py

# CNN + LSTM
python CNN_LSTMs.py

# ViT + MuRIL
python Vit_Muril.py

# ViT + TCN
python Vit_Tcn.py

Requirements

torch
torchvision
transformers
Pillow
pandas
numpy
scikit-learn

Install with:

bashpip install torch torchvision transformers Pillow pandas numpy scikit-learn

Training Configuration

ModelOptimizerLRBatchEpochsSchedulerCNN+LSTMAdam1e-43280StepLRViT+TCNAdamW5e-532100Cosine+WUResNet-50+mBERTAdamWvis:1e-4, txt:2e-53250Cosine+WUCLIPAdamWadaptive8100—ViT+MuRILAdamWadaptive8100Cosine+WU

All models trained on 5 seeds: {42, 123, 456, 789, 2024}

Citation

bibtex@misc{khatiwada2026nepooc,
  author    = {Khatiwada, Sanjeev},
  title     = {NepOOC-M: Bilingual Nepali-English Benchmark and Comparative
               Analysis of Multimodal Architectures for OOC Detection},
  year      = {2026},
  url       = {https://github.com/SanjeevKCodes/nepooc},
  note      = {Dataset: https://huggingface.co/datasets/theonlysanjeev/nepal-ooc-misinformation}
}

License

This code is released under CC BY-NC 4.0 (research-only, non-commercial).
See LICENSE for details.

Contact

Sanjeev Khatiwada — skhatiwada558@gmail.com — Kathmandu, Nepal

