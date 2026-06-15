"""
nepooc_unified_final.py
========================
ONE script. Runs ALL experiments for the NepOOC paper on Kaggle.

Covers EXACTLY:
  • All 7 model types: CNN+LSTM, ViT+TCN, ResNet-50+mBERT, CLIP,
                       ViT+MuRIL, text-only mBERT, text-only MuRIL
  • 5 seeds × 4 fractions × 3 modality modes (multimodal / text-only / image-only)
  • Tables VI–XV as in the paper

PLUS the 3 missing experiments:
  1. AUC for text-only mBERT and text-only MuRIL  (softmax probs always saved)
  2. McNemar's test: mBERT-text vs ResNet+mBERT,
                    ViT+MuRIL vs ResNet+mBERT,
                    MuRIL-text vs mBERT-text
  3. Leakage validation: cluster split vs random split for all 5 models

ARCHITECTURE SOURCE:
  Every model class is copied VERBATIM from the original single-model scripts
  (CNN_LSTMs.py, ViT_Tcn.py, Resnet_mBert.py, clip.py, Vit_Muril.py).
  No architectural drift.

BUG FIXED:
  CNN+LSTM StepLR step_size was 30 in the original; paper Table VI says 10.
  Fixed here.

KAGGLE SETUP:
  GPU  : T4 x2  (or P100)
  Internet: ON
  Add both datasets as input:
    • nepOOC CSVs  → /kaggle/input/datasets/amanlamichhane1234/nepooc-datset
    • NepOOC images → /kaggle/input/datasets/sanjeevkhatiwada/nepali-ooc-images
"""

# ============================================================
# 0.  INSTALL & IMPORTS
# ============================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.40.0", "timm==0.9.16",
                "statsmodels", "openai-clip"], check=True)

import os, copy, random, time, json, warnings
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR, StepLR
)

import torchvision.transforms as T
import torchvision.models as tvm
from torchvision.models import ResNet50_Weights

import timm
import clip as openai_clip

from transformers import (
    BertTokenizer, BertModel,
    AutoTokenizer, AutoModel,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit
from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings("ignore")

# ============================================================
# 1.  PATHS  — adjust dataset slugs if yours differ
# ============================================================
CSV_DIR = Path("/kaggle/input/datasets/amanlamichhane1234/nepooc-datset")
IMG_DIR = Path("/kaggle/input/datasets/sanjeevkhatiwada/nepali-ooc-images")
OUT_DIR = Path("/kaggle/working/nepooc_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2.  GLOBAL CONSTANTS  (paper values)
# ============================================================
SEEDS     = [42, 123, 456, 789, 2024]
FRACTIONS = [0.25, 0.50, 0.75, 1.0]

TYPO_KEYS = ["Fabricated", "Miscaptioned",
             "Temporal_Mismatch", "Geographic_Mismatch", "Identity_Mismatch"]
TYPO_DISPLAY = {
    "Fabricated":          "Fabricated",
    "Miscaptioned":        "Miscaptioned",
    "Temporal_Mismatch":   "Temporal mismatch",
    "Geographic_Mismatch": "Geographic mismatch",
    "Identity_Mismatch":   "Identity mismatch",
    "pristine":            "Pristine",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# 3.  REPRODUCIBILITY
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ============================================================
# 4.  DATA LOADING
# ============================================================
def get_image_path(post_id: str) -> Optional[str]:
    """Try exact match first, then recursive glob. Same logic as all originals."""
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = IMG_DIR / f"{post_id}{ext}"
        if p.exists():
            return str(p)
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        hits = list(IMG_DIR.rglob(f"*{post_id}*{ext}"))
        if hits:
            return str(hits[0])
    return None


def load_dataframes():
    """Load the 3 pre-split CSVs, add image_path, drop missing images."""
    dfs = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(CSV_DIR / f"nepOOC_{split}.csv")
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(int)
        df["image_path"] = df["post_id"].apply(get_image_path)
        n_before = len(df)
        df = df[df["image_path"].notna()].reset_index(drop=True)
        print(f"  {split}: {len(df)}/{n_before} rows have images  "
              f"(OOC={df['label'].sum()}, Pristine={(df['label']==0).sum()})")
        dfs[split] = df
    return dfs["train"], dfs["val"], dfs["test"]


def subsample(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Stratified subsample preserving 50/50 balance — used by all models."""
    if fraction >= 1.0:
        return df.reset_index(drop=True)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=seed)
    idx, _ = next(sss.split(df, df["label"]))
    return df.iloc[idx].reset_index(drop=True)


def mtype_from_row(row) -> str:
    v = row.get("misinformation_type", None)
    return "pristine" if (v is None or pd.isna(v)) else str(v)


# ============================================================
# 5.  COMMON IMAGE TRANSFORMS
# ============================================================
TRANSFORM_224 = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

TRANSFORM_448 = T.Compose([
    T.Resize(448, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(448),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_BLACK_224 = torch.zeros(3, 224, 224)
_BLACK_448 = torch.zeros(3, 448, 448)


def open_image(path: str, transform, black_fallback: torch.Tensor) -> torch.Tensor:
    try:
        return transform(Image.open(path).convert("RGB"))
    except Exception:
        return black_fallback.clone()


# ============================================================
# 6.  DATASET CLASSES  (one per model family to match originals)
# ============================================================

# ---- 6A: Standard dataset (CNN+LSTM, ResNet+mBERT, ViT+TCN, ViT+MuRIL) ----
class NepOOCStandard(Dataset):
    """
    Returns: image tensor, input_ids, attention_mask, label, post_id, mtype
    Used by: CNN+LSTM, ResNet+mBERT, ViT+TCN (text-only/image-only ablations too)
    """
    def __init__(self, df: pd.DataFrame, tokenizer, transform, img_size=224):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.black     = _BLACK_224 if img_size == 224 else _BLACK_448

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        img     = open_image(row["image_path"], self.transform, self.black)
        caption = str(row["caption"]) if pd.notna(row["caption"]) else ""
        enc     = self.tokenizer(caption, max_length=128, padding="max_length",
                                 truncation=True, return_tensors="pt")
        return {
            "image":               img,
            "input_ids":           enc["input_ids"].squeeze(0),
            "attention_mask":      enc["attention_mask"].squeeze(0),
            "label":               torch.tensor(int(row["label"]), dtype=torch.long),
            "post_id":             str(row["post_id"]),
            "misinformation_type": mtype_from_row(row),
        }


# ---- 6B: CLIP dataset -------------------------------------------------------
class NepOOCClip(Dataset):
    """Returns: image tensor (CLIP preprocess), text token tensor, label, …"""
    def __init__(self, df: pd.DataFrame, clip_preprocess):
        self.df        = df.reset_index(drop=True)
        self.preprocess = clip_preprocess

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        caption = str(row["caption"]) if pd.notna(row["caption"]) else ""
        try:
            img = self.preprocess(Image.open(row["image_path"]).convert("RGB"))
        except Exception:
            img = self.preprocess(Image.new("RGB", (224, 224), (0, 0, 0)))
        text  = openai_clip.tokenize([caption], truncate=True).squeeze(0)
        mtype = mtype_from_row(row)
        return {
            "image":               img,
            "text":                text,
            "label":               torch.tensor(int(row["label"]), dtype=torch.long),
            "post_id":             str(row["post_id"]),
            "misinformation_type": mtype,
        }


# ============================================================
# 7.  MODEL ARCHITECTURES  (verbatim from originals)
# ============================================================

# ---- 7A: CNN+LSTM -----------------------------------------------------------
VOCAB_SIZE_MBERT = 119_547

class CNN5Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(512 * 4 * 4, 512)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        return self.drop(torch.relu(self.proj(self.features(x).view(x.size(0), -1))))


class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE_MBERT, 128, padding_idx=0)
        self.lstm  = nn.LSTM(128, 256, batch_first=True, bidirectional=True)
        self.proj  = nn.Linear(256 * 2, 512)
        self.drop  = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        _, (h, _) = self.lstm(self.embed(input_ids))
        return self.drop(torch.relu(self.proj(torch.cat([h[0], h[1]], dim=1))))


class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = CNN5Layer()
        self.lstm = LSTMEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 2))

    def forward(self, imgs, input_ids, attn_mask):
        return self.classifier(torch.cat([self.cnn(imgs),
                                          self.lstm(input_ids, attn_mask)], dim=1))


class CNNLSTMTextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = LSTMEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 2))

    def forward(self, imgs, input_ids, attn_mask):
        return self.classifier(self.lstm(input_ids, attn_mask))


class CNNLSTMImageOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN5Layer()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 2))

    def forward(self, imgs, input_ids, attn_mask):
        return self.classifier(self.cnn(imgs))


# ---- 7B: ViT+TCN ------------------------------------------------------------
class DilatedResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation):
        super().__init__()
        padding   = (kernel - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        skip = x if self.skip is None else self.skip(x)
        return self.relu(self.bn(self.conv(x)) + skip)


class TCNEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE_MBERT, embedding_dim=128,
                 hidden_dim=256, output_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.block1    = DilatedResidualBlock(embedding_dim, hidden_dim, 3, dilation=1)
        self.block2    = DilatedResidualBlock(hidden_dim,    hidden_dim, 3, dilation=2)
        self.block3    = DilatedResidualBlock(hidden_dim,    hidden_dim, 3, dilation=4)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        x    = self.embedding(input_ids).transpose(1, 2)
        x    = self.block3(self.block2(self.block1(x)))
        mask = attention_mask.unsqueeze(1).float()
        x    = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        return self.projection(x)


class ViTTCNModel(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit        = vit_model
        self.tcn        = TCNEncoder()
        self.query_proj = nn.Linear(768, 64)
        self.key_proj   = nn.Linear(768, 64)
        self.value_proj = nn.Linear(768, 768)
        self.layer_norm = nn.LayerNorm(768)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, images, input_ids, attention_mask):
        patch = self.vit.forward_features(images)[:, 1:, :]   # (B,196,768)
        t     = self.tcn(input_ids, attention_mask)            # (B,768)
        Q = self.query_proj(t).unsqueeze(1)                    # (B,1,64)
        K = self.key_proj(patch)                               # (B,196,64)
        V = self.value_proj(patch)                             # (B,196,768)
        w = F.softmax(Q @ K.transpose(-2, -1) / (64 ** 0.5), dim=-1)
        fused = self.layer_norm((w @ V).squeeze(1) + t)
        return self.classifier(fused)


class ViTTCNTextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.tcn = TCNEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, images, input_ids, attention_mask):
        return self.classifier(self.tcn(input_ids, attention_mask))


class ViTTCNImageOnly(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.vit = vit_model
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, images, input_ids, attention_mask):
        return self.classifier(self.vit.forward_features(images)[:, 0, :])


# ---- 7C: ResNet-50 + mBERT --------------------------------------------------
class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone      = tvm.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj     = nn.Linear(2048, 768)
        self.drop     = nn.Dropout(0.1)

    def forward(self, x):
        return self.drop(torch.relu(self.proj(self.features(x).flatten(1))))


class MBERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.drop = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        return self.drop(
            self.bert(input_ids=input_ids,
                      attention_mask=attention_mask).last_hidden_state[:, 0])


class ResNetMBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet     = ResNetEncoder()
        self.mbert      = MBERTEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, imgs, input_ids, attn_mask):
        return self.classifier(
            torch.cat([self.resnet(imgs), self.mbert(input_ids, attn_mask)], dim=1))


class ResNetMBERTTextOnly(nn.Module):
    """Standalone mBERT text-only (Table VI dedicated baseline)."""
    def __init__(self):
        super().__init__()
        self.mbert      = MBERTEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, imgs, input_ids, attn_mask):
        return self.classifier(self.mbert(input_ids, attn_mask))


class ResNetMBERTImageOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet     = ResNetEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, imgs, input_ids, attn_mask):
        return self.classifier(self.resnet(imgs))


# ---- 7D: CLIP ---------------------------------------------------------------
class CLIPMultimodal(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512 + 1 + 512, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 2))

    def forward(self, images, texts):
        v = F.normalize(self.clip.encode_image(images).float(), dim=-1)
        t = F.normalize(self.clip.encode_text(texts).float(),  dim=-1)
        return self.classifier(torch.cat([v, t, (v * t).sum(-1, keepdim=True), v - t], dim=1))


class CLIPTextOnly(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, images, texts):
        return self.classifier(
            F.normalize(self.clip.encode_text(texts).float(), dim=-1))


class CLIPImageOnly(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, images, texts):
        return self.classifier(
            F.normalize(self.clip.encode_image(images).float(), dim=-1))


# ---- 7E: ViT+MuRIL ----------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear
        d_in, d_out = linear.weight.shape[1], linear.weight.shape[0]
        for p in self.linear.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale  = alpha / rank

    def forward(self, x):
        return self.linear(x) + self.scale * (x @ self.lora_A.T @ self.lora_B.T)


def apply_lora_to_muril(model: nn.Module, rank=8, alpha=16):
    for layer in model.encoder.layer:
        a = layer.attention.self
        a.query = LoRALinear(a.query, rank=rank, alpha=alpha)
        a.value = LoRALinear(a.value, rank=rank, alpha=alpha)
    for name, p in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            p.requires_grad = False


def build_vit_448(pretrained=True) -> nn.Module:
    """ViT-B/16 with positional embedding interpolated to 448×448 (784 patches)."""
    vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained)
    vit.patch_embed.img_size   = (448, 448)
    vit.patch_embed.grid_size  = (28, 28)
    vit.patch_embed.num_patches = 784
    with torch.no_grad():
        pe  = vit.pos_embed                       # (1, 197, 768)
        cls, patches = pe[:, :1], pe[:, 1:]
        patches = (patches.reshape(1, 14, 14, 768)
                          .permute(0, 3, 1, 2).float())
        patches = F.interpolate(patches, size=(28, 28),
                                mode="bicubic", align_corners=False)
        patches = patches.permute(0, 2, 3, 1).reshape(1, 784, 768)
        vit.pos_embed = nn.Parameter(torch.cat([cls, patches], dim=1))
    for p in vit.parameters():
        p.requires_grad = False
    return vit


class TextToImageCrossAttention(nn.Module):
    def __init__(self, d_model=768, dk=64):
        super().__init__()
        self.dk    = dk
        self.W_Q   = nn.Linear(d_model, dk, bias=False)
        self.W_K   = nn.Linear(d_model, dk, bias=False)
        self.W_V   = nn.Linear(d_model, dk, bias=False)
        self.W_out = nn.Linear(dk, d_model, bias=False)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, text_q, v_patches):
        Q = self.W_Q(text_q).unsqueeze(1)                             # (B,1,dk)
        K = self.W_K(v_patches)                                       # (B,784,dk)
        V = self.W_V(v_patches)                                       # (B,784,dk)
        attn = F.softmax(Q @ K.transpose(1, 2) / (self.dk ** 0.5), dim=-1)
        out  = self.W_out(( attn @ V).squeeze(1))                     # (B,d_model)
        return self.norm(out + text_q)


class ViTMuRILMultimodal(nn.Module):
    def __init__(self, vit, muril):
        super().__init__()
        self.vit        = vit
        self.muril      = muril
        self.cross_attn = TextToImageCrossAttention()
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, images, input_ids, attention_mask):
        patches  = self.vit.forward_features(images)[:, 1:, :]        # (B,784,768)
        text_cls = self.muril(input_ids=input_ids,
                              attention_mask=attention_mask
                              ).last_hidden_state[:, 0]                # (B,768)
        return self.classifier(self.cross_attn(text_cls, patches))


class ViTMuRILTextOnly(nn.Module):
    def __init__(self, muril):
        super().__init__()
        self.muril      = muril
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, images, input_ids, attention_mask):
        cls = self.muril(input_ids=input_ids,
                         attention_mask=attention_mask
                         ).last_hidden_state[:, 0]
        return self.classifier(cls)


class ViTMuRILImageOnly(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit        = vit
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, images, input_ids, attention_mask):
        return self.classifier(self.vit.forward_features(images)[:, 0])


def build_vit_muril_model(model_type: str, vit_base, muril_base) -> nn.Module:
    muril = copy.deepcopy(muril_base)
    vit   = copy.deepcopy(vit_base)
    if model_type in ("multimodal", "text_only"):
        apply_lora_to_muril(muril)
    if model_type == "multimodal":
        return ViTMuRILMultimodal(vit, muril)
    elif model_type == "text_only":
        return ViTMuRILTextOnly(muril)
    else:
        return ViTMuRILImageOnly(vit)


# ---- 7F: Standalone text-only mBERT (dedicated baseline in Table VI) --------
class StandaloneMBERT(nn.Module):
    """
    Pure text-only mBERT — the dedicated baseline in Table VI.
    Identical to ResNetMBERTTextOnly but named separately for clarity.
    Saves softmax probs → enables AUC computation (fixing paper gap).
    """
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.drop = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, input_ids, attention_mask, images=None):
        cls = self.drop(
            self.bert(input_ids=input_ids,
                      attention_mask=attention_mask).last_hidden_state[:, 0])
        return self.classifier(cls)


# ---- 7G: Standalone text-only MuRIL (dedicated baseline in Table VI) --------
class StandaloneMuRIL(nn.Module):
    """
    Pure text-only MuRIL — the other dedicated baseline in Table VI.
    Saves softmax probs → enables AUC computation (fixing paper gap).
    """
    def __init__(self, muril_base):
        super().__init__()
        self.muril = copy.deepcopy(muril_base)
        apply_lora_to_muril(self.muril)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 2))

    def forward(self, input_ids, attention_mask, images=None):
        cls = self.muril(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.classifier(cls)


# ============================================================
# 8.  EVALUATE FUNCTION  (unified — works for all model families)
# ============================================================
def evaluate(model, loader, model_family: str,
             return_details: bool = False, criterion=None) -> dict:
    """
    model_family: "standard" | "clip" | "text_only_standalone"

    Always returns:
      acc, f1_macro, auc, preds (np), probs (np), labels (np),
      precision_ooc, recall_ooc, f1_ooc, confusion_matrix,
      post_ids, types   (empty lists unless return_details=True)
    """
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    all_post_ids, all_types = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(DEVICE)

            if model_family == "clip":
                imgs   = batch["image"].to(DEVICE)
                texts  = batch["text"].to(DEVICE)
                logits = model(imgs, texts)
            elif model_family == "text_only_standalone":
                ids    = batch["input_ids"].to(DEVICE)
                mask   = batch["attention_mask"].to(DEVICE)
                logits = model(ids, mask)
            else:  # "standard" (CNN+LSTM, ResNet+mBERT, ViT+TCN, ViT+MuRIL + ablations)
                imgs   = batch["image"].to(DEVICE)
                ids    = batch["input_ids"].to(DEVICE)
                mask   = batch["attention_mask"].to(DEVICE)
                logits = model(imgs, ids, mask)

            if criterion is not None:
                total_loss += criterion(logits, labels).item()

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            if return_details:
                all_post_ids.extend(batch.get("post_id", [""] * len(labels)))
                all_types.extend(batch.get("misinformation_type", [""] * len(labels)))

    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = 0.5

    return {
        "acc":            float(accuracy_score(all_labels, all_preds)),
        "f1_macro":       float(f1_score(all_labels, all_preds, average="macro")),
        "auc":            auc,                  # ← now always computed, fixes paper gap
        "precision_ooc":  float(precision_score(all_labels, all_preds, pos_label=1, zero_division=0)),
        "recall_ooc":     float(recall_score(   all_labels, all_preds, pos_label=1, zero_division=0)),
        "f1_ooc":         float(f1_score(       all_labels, all_preds, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist(),
        "preds":          all_preds,
        "probs":          all_probs,
        "labels":         all_labels,
        "post_ids":       all_post_ids,
        "types":          all_types,
        "loss":           total_loss / max(len(loader), 1),
    }


# ============================================================
# 9.  SCHEDULER BUILDERS
# ============================================================
def make_cosine_warmup(optimizer, total_steps: int, warmup_frac=0.10):
    warmup = int(total_steps * warmup_frac)
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-8, end_factor=1.0,
                     total_iters=max(1, warmup)),
            CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup),
                              eta_min=1e-7),
        ],
        milestones=[max(1, warmup)],
    )


# ============================================================
# 10. TRAINING LOOPS  (one per model family — mirrors originals)
# ============================================================

# ── 10A: CNN+LSTM ─────────────────────────────────────────────────────────────
def train_cnn_lstm(model_cls, seed, fraction,
                   train_df, val_df, test_df, mbert_tok,
                   model_name="multimodal"):
    """
    Paper Table VI: Adam, LR=1e-4, WD=1e-5, BS=32, EP=80, PAT=10,
                    StepLR step_size=10 gamma=0.5   ← FIXED (was 30 in original)
    """
    seed_everything(seed)
    sub = subsample(train_df, fraction, seed)

    ds_kwargs = dict(tokenizer=mbert_tok, transform=TRANSFORM_224, img_size=224)
    train_ds = NepOOCStandard(sub,      **ds_kwargs)
    val_ds   = NepOOCStandard(val_df,   **ds_kwargs)
    test_ds  = NepOOCStandard(test_df,  **ds_kwargs)

    dl_kw = dict(batch_size=32, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kw)

    model     = model_cls().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)   # FIXED from 30→10

    best_val_f1 = 0.0
    best_state  = {k: v.cpu() for k, v in model.state_dict().items()}
    no_improve  = 0

    for epoch in range(1, 81):
        model.train()
        for batch in train_loader:
            imgs, ids, mask, labels = (batch["image"].to(DEVICE),
                                       batch["input_ids"].to(DEVICE),
                                       batch["attention_mask"].to(DEVICE),
                                       batch["label"].to(DEVICE))
            optimizer.zero_grad()
            criterion(model(imgs, ids, mask), labels).backward()
            optimizer.step()
        scheduler.step()

        val_m = evaluate(model, val_loader, "standard")
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            best_state  = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                break

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, "standard", return_details=True)
    test_m.update(dict(seed=seed, fraction=fraction, best_val_f1=best_val_f1,
                       model=model_name))
    return test_m


# ── 10B: ViT+TCN ──────────────────────────────────────────────────────────────
def train_vit_tcn(model_cls, vit_model, seed, fraction,
                  train_df, val_df, test_df, mbert_tok,
                  model_name="multimodal"):
    """
    Paper Table VI: AdamW, LR=5e-5, WD=1e-4, BS=32, EP=100,
                    PAT=10, CosineWU
    """
    seed_everything(seed)
    sub = subsample(train_df, fraction, seed)

    ds_kwargs = dict(tokenizer=mbert_tok, transform=TRANSFORM_224, img_size=224)
    train_ds = NepOOCStandard(sub,     **ds_kwargs)
    val_ds   = NepOOCStandard(val_df,  **ds_kwargs)
    test_ds  = NepOOCStandard(test_df, **ds_kwargs)

    dl_kw = dict(batch_size=32, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kw)

    model     = model_cls(vit_model).to(DEVICE) if vit_model else model_cls().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = make_cosine_warmup(optimizer, 100 * len(train_loader))

    best_val_f1, best_state, no_improve = 0.0, None, 0
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            imgs, ids, mask, labels = (batch["image"].to(DEVICE),
                                       batch["input_ids"].to(DEVICE),
                                       batch["attention_mask"].to(DEVICE),
                                       batch["label"].to(DEVICE))
            optimizer.zero_grad()
            criterion(model(imgs, ids, mask), labels).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        val_m = evaluate(model, val_loader, "standard")
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            best_state  = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                break

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, "standard", return_details=True)
    test_m.update(dict(seed=seed, fraction=fraction, best_val_f1=best_val_f1,
                       model=model_name))
    return test_m


# ── 10C: ResNet-50 + mBERT ────────────────────────────────────────────────────
def _build_resnet_optimizer(model):
    bert_ids     = {id(p) for p in model.mbert.parameters()} if hasattr(model, "mbert") else set()
    text_params  = [p for p in model.parameters() if id(p) in bert_ids]
    other_params = [p for p in model.parameters() if id(p) not in bert_ids]
    return AdamW([{"params": text_params,  "lr": 2e-5},
                  {"params": other_params, "lr": 1e-4}], weight_decay=1e-4)


def train_resnet_mbert(model_cls, seed, fraction,
                       train_df, val_df, test_df, mbert_tok,
                       model_name="multimodal"):
    """
    Paper Table VI: AdamW, LR_vis=1e-4 LR_txt=2e-5, WD=1e-4,
                    BS=32, EP=50, PAT=10, CosineWU
    """
    seed_everything(seed)
    sub = subsample(train_df, fraction, seed)

    ds_kwargs = dict(tokenizer=mbert_tok, transform=TRANSFORM_224, img_size=224)
    train_ds = NepOOCStandard(sub,     **ds_kwargs)
    val_ds   = NepOOCStandard(val_df,  **ds_kwargs)
    test_ds  = NepOOCStandard(test_df, **ds_kwargs)

    dl_kw = dict(batch_size=32, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kw)

    model     = model_cls().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_resnet_optimizer(model)
    total_steps = 50 * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.10 * total_steps),
        num_training_steps=total_steps)

    best_val_f1, best_state, no_improve = 0.0, None, 0
    for epoch in range(50):
        model.train()
        for batch in train_loader:
            imgs, ids, mask, labels = (batch["image"].to(DEVICE),
                                       batch["input_ids"].to(DEVICE),
                                       batch["attention_mask"].to(DEVICE),
                                       batch["label"].to(DEVICE))
            optimizer.zero_grad()
            criterion(model(imgs, ids, mask), labels).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        val_m = evaluate(model, val_loader, "standard")
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            best_state  = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                break

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, "standard", return_details=True)
    test_m.update(dict(seed=seed, fraction=fraction, best_val_f1=best_val_f1,
                       model=model_name))
    return test_m


# ── 10D: CLIP ─────────────────────────────────────────────────────────────────
def train_clip_model(model_cls, seed, fraction,
                     train_df, val_df, test_df, clip_preprocess,
                     model_name="multimodal"):
    """
    Paper Table VI: AdamW, LR adaptive (1e-4 if frac≤0.5 else 5e-5),
                    WD=0.05, BS=8, EP=100, PAT=12, no scheduler
    """
    seed_everything(seed)
    sub = subsample(train_df, fraction, seed)

    train_ds = NepOOCClip(sub,     clip_preprocess)
    val_ds   = NepOOCClip(val_df,  clip_preprocess)
    test_ds  = NepOOCClip(test_df, clip_preprocess)

    dl_kw = dict(batch_size=8, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kw)

    clip_model, _ = openai_clip.load("ViT-B/32", device=DEVICE)
    model     = model_cls(clip_model).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    lr        = 1e-4 if fraction <= 0.50 else 5e-5
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    best_val_f1, best_state, no_improve = 0.0, None, 0
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            imgs, texts, labels = (batch["image"].to(DEVICE),
                                   batch["text"].to(DEVICE),
                                   batch["label"].to(DEVICE))
            optimizer.zero_grad()
            criterion(model(imgs, texts), labels).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_m = evaluate(model, val_loader, "clip")
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            best_state  = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= 12:
                break

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, "clip", return_details=True)
    test_m.update(dict(seed=seed, fraction=fraction, best_val_f1=best_val_f1,
                       model=model_name))
    return test_m


# ── 10E: ViT+MuRIL ────────────────────────────────────────────────────────────
def train_vit_muril(model_type, vit_base, muril_base, seed, fraction,
                    train_df, val_df, test_df, muril_tok):
    """
    Paper Table VI: AdamW, LR adaptive (1e-4 if frac≤0.5 else 5e-5),
                    WD=0.05, BS=8, EP=100, PAT=12, CosineWU,
                    label_smoothing=0.1
    """
    seed_everything(seed)
    sub = subsample(train_df, fraction, seed)

    ds_kwargs = dict(tokenizer=muril_tok, transform=TRANSFORM_448, img_size=448)
    train_ds = NepOOCStandard(sub,     **ds_kwargs)
    val_ds   = NepOOCStandard(val_df,  **ds_kwargs)
    test_ds  = NepOOCStandard(test_df, **ds_kwargs)

    dl_kw = dict(batch_size=8, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kw)

    model     = build_vit_muril_model(model_type, vit_base, muril_base).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr        = 1e-4 if fraction <= 0.50 else 5e-5
    optimizer = AdamW(trainable, lr=lr, weight_decay=0.05)
    scheduler = make_cosine_warmup(optimizer, 100 * len(train_loader))

    best_val_f1, best_state, no_improve = 0.0, None, 0
    for epoch in range(100):
        model.train()
        for batch in train_loader:
            imgs, ids, mask, labels = (batch["image"].to(DEVICE),
                                       batch["input_ids"].to(DEVICE),
                                       batch["attention_mask"].to(DEVICE),
                                       batch["label"].to(DEVICE))
            optimizer.zero_grad()
            criterion(model(imgs, ids, mask), labels).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()

        val_m = evaluate(model, val_loader, "standard")
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            best_state  = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= 12:
                break

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, "standard", return_details=True)
    test_m.update(dict(seed=seed, fraction=fraction, best_val_f1=best_val_f1,
                       model=model_type))
    del model; torch.cuda.empty_cache()
    return test_m


# ── 10F: Standalone text-only mBERT ──────────────────────────────────────────
def train_standalone_mbert(seed, fraction, train_df, val_df, test_df, mbert_tok):
    """
    Dedicated text-only mBERT baseline (Table VI).
    Uses same AdamW / CosineWU as ResNet+mBERT text group.
    AUC is always computed from softmax probs — fixes the paper gap.
    """
    seed_everything(seed)
    sub = subsample(train_df, fraction, seed)

    ds_kwargs = dict(tokenizer=mbert_tok, transform=TRANSFORM_224, img_size=224)
    train_ds = NepOOCStandard(sub,     **ds_kwargs)
    val_ds   = NepOOCStandard(val_df,  **ds_kwargs)
    test_ds  = NepOOCStandard(test_df, **ds_kwargs)

    dl_kw = dict(batch_size=32, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kw)

    model     = StandaloneMBERT().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    total_steps = 50 * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.10 * total_steps),
        num_training_steps=total_steps)

    best_val_f1, best_state, no_improve = 0.0, None, 0
    for epoch in range(50):
        model.train()
        for batch in train_loader:
            ids, mask, labels = (batch["input_ids"].to(DEVICE),
                                 batch["attention_mask"].to(DEVICE),
                                 batch["label"].to(DEVICE))
            optimizer.zero_grad()
            criterion(model(ids, mask), labels).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

        val_m = evaluate(model, val_loader, "text_only_standalone")
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            best_state  = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                break

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, "text_only_standalone")
    test_m.update(dict(seed=seed, fraction=fraction, best_val_f1=best_val_f1,
                       model="text_only_mbert"))
    return test_m


# ── 10G: Standalone text-only MuRIL ──────────────────────────────────────────
def train_standalone_muril(muril_base, seed, fraction,
                           train_df, val_df, test_df, muril_tok):
    """
    Dedicated text-only MuRIL baseline (Table VI).
    AUC always computed — fixes paper gap.
    """
    seed_everything(seed)
    sub = subsample(train_df, fraction, seed)

    ds_kwargs = dict(tokenizer=muril_tok, transform=TRANSFORM_448, img_size=448)
    train_ds = NepOOCStandard(sub,     **ds_kwargs)
    val_ds   = NepOOCStandard(val_df,  **ds_kwargs)
    test_ds  = NepOOCStandard(test_df, **ds_kwargs)

    dl_kw = dict(batch_size=32, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kw)

    model     = StandaloneMuRIL(muril_base).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=2e-5, weight_decay=0.05)
    total_steps = 50 * len(train_loader)
    scheduler = make_cosine_warmup(optimizer, total_steps)

    best_val_f1, best_state, no_improve = 0.0, None, 0
    for epoch in range(50):
        model.train()
        for batch in train_loader:
            ids, mask, labels = (batch["input_ids"].to(DEVICE),
                                 batch["attention_mask"].to(DEVICE),
                                 batch["label"].to(DEVICE))
            optimizer.zero_grad()
            criterion(model(ids, mask), labels).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step()

        val_m = evaluate(model, val_loader, "text_only_standalone")
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1 = val_m["f1_macro"]
            best_state  = {k: v.cpu() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                break

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, "text_only_standalone")
    test_m.update(dict(seed=seed, fraction=fraction, best_val_f1=best_val_f1,
                       model="text_only_muril"))
    del model; torch.cuda.empty_cache()
    return test_m


# ============================================================
# 11. STATISTICS HELPERS
# ============================================================
def compute_stats(results: list, fraction=1.0) -> Optional[dict]:
    sub = [r for r in results if r["fraction"] == fraction]
    if not sub:
        return None
    accs  = [r["acc"]      for r in sub]
    f1s   = [r["f1_macro"] for r in sub]
    aucs  = [r["auc"]      for r in sub]
    precs = [r["precision_ooc"] for r in sub]
    recs  = [r["recall_ooc"]    for r in sub]
    cms   = np.array([r["confusion_matrix"] for r in sub])
    return {
        "acc_mean":  np.mean(accs),   "acc_std":  np.std(accs,  ddof=1),
        "f1_mean":   np.mean(f1s),    "f1_std":   np.std(f1s,   ddof=1),
        "auc_mean":  np.mean(aucs),   "auc_std":  np.std(aucs,  ddof=1),
        "prec_mean": np.mean(precs),  "prec_std": np.std(precs, ddof=1),
        "rec_mean":  np.mean(recs),   "rec_std":  np.std(recs,  ddof=1),
        "confusion_matrix": cms.mean(axis=0).tolist(),
        "n_seeds":   len(sub),
    }


def print_table_vi(name: str, results: list):
    s = compute_stats(results, fraction=1.0)
    if s is None:
        print(f"  {name}: no results at 100%")
        return
    print(f"  {name:<25s}  "
          f"Acc={s['acc_mean']*100:.2f}±{s['acc_std']*100:.2f}  "
          f"F1={s['f1_mean']*100:.2f}±{s['f1_std']*100:.2f}  "
          f"AUC={s['auc_mean']:.4f}±{s['auc_std']:.4f}")


def per_typology_f1_across_seeds(results_100pct: list) -> dict:
    """
    Compute per-typology F1 mean ± std (ddof=1) across seeds.
    results_100pct: list of result dicts at fraction=1.0.
    """
    typo_seed_f1 = defaultdict(list)
    for res in results_100pct:
        preds  = np.array(res["preds"])
        labels = np.array(res["labels"])
        types  = np.array(res["types"])
        ooc_mask = labels == 1
        for typo in TYPO_KEYS:
            mask = (types == typo) & ooc_mask
            if mask.sum() == 0:
                continue
            typo_preds  = preds[mask]
            typo_labels = np.ones(mask.sum(), dtype=int)
            f1 = f1_score(typo_labels, typo_preds, average="binary",
                          pos_label=1, zero_division=0)
            typo_seed_f1[typo].append(f1)

    out = {}
    for typo, vals in typo_seed_f1.items():
        out[typo] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n":    len(vals),
        }
    return out


# ============================================================
# 12. McNEMAR'S TEST
# ============================================================
def run_mcnemar(results_a: list, results_b: list,
                name_a: str, name_b: str, fraction=1.0):
    """
    Per-seed McNemar's test comparing two model result lists.
    results_a / results_b: full result dicts at the given fraction.
    """
    print(f"\n  McNemar: {name_a} vs {name_b}")
    sub_a = [r for r in results_a if r["fraction"] == fraction]
    sub_b = [r for r in results_b if r["fraction"] == fraction]
    assert len(sub_a) == len(sub_b) == 5, \
        f"Expected 5 seeds each, got {len(sub_a)} and {len(sub_b)}"

    pvals = []
    for i, (ra, rb) in enumerate(zip(sub_a, sub_b)):
        labels = np.array(ra["labels"])
        pa = (np.array(ra["preds"]) == labels).astype(int)
        pb = (np.array(rb["preds"]) == labels).astype(int)
        n11 = int(((pa == 1) & (pb == 1)).sum())
        n10 = int(((pa == 1) & (pb == 0)).sum())
        n01 = int(((pa == 0) & (pb == 1)).sum())
        n00 = int(((pa == 0) & (pb == 0)).sum())
        table  = np.array([[n11, n10], [n01, n00]])
        result = mcnemar(table, exact=True)
        pvals.append(result.pvalue)
        sig = "SIGNIFICANT" if result.pvalue < 0.05 else "not significant"
        print(f"    Seed {SEEDS[i]}: n10={n10}  n01={n01}  "
              f"p={result.pvalue:.4f}  ({sig})")

    pvals = np.array(pvals)
    print(f"    → Mean p={pvals.mean():.4f}  "
          f"All p>0.05: {(pvals > 0.05).all()}")
    return pvals


# ============================================================
# 13. EVENT-CLUSTER LEAKAGE VALIDATION
# ============================================================
def parse_date(date_str: str):
    """Parse DD/MM/YYYY → (year, month). Matches your posted_date format."""
    try:
        parts = str(date_str).strip().split("/")
        month = int(parts[1]) if len(parts) > 1 else 0
        year  = int(parts[2]) if len(parts) > 2 else None
        return (year, month)
    except Exception:
        return (None, 0)


@torch.no_grad()
def extract_resnet_embeddings(df: pd.DataFrame) -> np.ndarray:
    """ResNet-50 pool5 embeddings (2048-d) for image clustering."""
    backbone = tvm.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity()
    backbone.eval().to(DEVICE)

    dataset = NepOOCStandard(df, tokenizer=BertTokenizer.from_pretrained(
        "bert-base-multilingual-cased"), transform=TRANSFORM_224)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=2, pin_memory=True)
    feats   = []
    for batch in loader:
        feats.append(backbone(batch["image"].to(DEVICE)).cpu().numpy())
    del backbone; torch.cuda.empty_cache()
    return np.vstack(feats)


def build_event_clusters(full_df: pd.DataFrame,
                         cos_threshold: float = 0.85) -> np.ndarray:
    """
    Three-stage clustering (paper Section III-F):
      Stage 1: temporal anchor from posted_date
      Stage 2: named-entity grouping from named_entities column
      Stage 3: ResNet-50 cosine similarity ≥ 0.85
    Returns integer cluster ID array, length = len(full_df).
    """
    n = len(full_df)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    # Stage 1+2: date + NER
    from collections import defaultdict
    groups = defaultdict(list)
    for i, row in full_df.reset_index(drop=True).iterrows():
        ym   = parse_date(row.get("posted_date", ""))
        ners = frozenset(
            e.strip().lower()
            for e in str(row.get("named_entities", "")).split(",")
            if e.strip() and e.strip().lower() != "nan"
        )
        groups[(ym, ners)].append(i)
    for members in groups.values():
        for j in range(1, len(members)):
            union(members[0], members[j])

    # Stage 3: image cosine similarity within temporal+NER groups
    print("  Extracting image embeddings for cluster analysis...")
    embs  = extract_resnet_embeddings(full_df.reset_index(drop=True))
    norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

    for members in groups.values():
        if len(members) < 2:
            continue
        sub  = norms[members]
        sims = sub @ sub.T
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                if sims[a, b] >= cos_threshold:
                    union(members[a], members[b])

    cluster_ids = np.array([find(i) for i in range(n)])
    unique      = {v: k for k, v in enumerate(sorted(set(cluster_ids)))}
    cluster_ids = np.array([unique[c] for c in cluster_ids])
    K = len(unique)
    print(f"  Found {K} event clusters (mean size {n/K:.1f} samples/cluster)")
    return cluster_ids


def make_cluster_split(full_df: pd.DataFrame, cluster_ids: np.ndarray,
                       test_frac=0.209, seed=42):
    """
    Assign whole clusters to train or test (paper cluster split).
    test_frac≈0.209 → ~228 test samples matching the random split.
    Returns train_df, test_df (no validation — callers use 10% of train).
    """
    rng      = np.random.RandomState(seed)
    clusters = sorted(set(cluster_ids))
    rng.shuffle(clusters)

    n_total    = len(full_df)
    n_test_tgt = int(n_total * test_frac)
    test_cls, n_test_now = [], 0
    train_cls = []
    for c in clusters:
        c_idx = np.where(cluster_ids == c)[0]
        if n_test_now < n_test_tgt:
            test_cls.append(c);  n_test_now += len(c_idx)
        else:
            train_cls.append(c)

    test_idx  = np.where(np.isin(cluster_ids, test_cls))[0]
    train_idx = np.where(np.isin(cluster_ids, train_cls))[0]
    print(f"  Cluster split → train={len(train_idx)}  test={len(test_idx)}")
    return (full_df.iloc[train_idx].reset_index(drop=True),
            full_df.iloc[test_idx].reset_index(drop=True))


# ============================================================
# 14. MAIN PIPELINE
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("  NepOOC UNIFIED EXPERIMENT")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_df, val_df, test_df = load_dataframes()
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # ── Load tokenisers / base models (once) ─────────────────────────────────
    print("\nLoading tokenisers and base models...")
    mbert_tok  = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    muril_tok  = AutoTokenizer.from_pretrained("google/muril-base-cased")
    muril_base = AutoModel.from_pretrained("google/muril-base-cased")
    for p in muril_base.parameters():
        p.requires_grad = False

    vit_tcn_base   = timm.create_model("vit_base_patch16_224",
                                       pretrained=True, num_classes=0)
    vit_muril_base = build_vit_448(pretrained=True)

    _, clip_preprocess = openai_clip.load("ViT-B/32", device=DEVICE)

    # ── Storage ──────────────────────────────────────────────────────────────
    all_results = {
        "cnn_lstm":         {"multimodal": [], "text_only": [], "image_only": []},
        "vit_tcn":          {"multimodal": [], "text_only": [], "image_only": []},
        "resnet_mbert":     {"multimodal": [], "text_only": [], "image_only": []},
        "clip":             {"multimodal": [], "text_only": [], "image_only": []},
        "vit_muril":        {"multimodal": [], "text_only": [], "image_only": []},
        "standalone_mbert": {"multimodal": []},   # "multimodal" slot holds text-only runs
        "standalone_muril": {"multimodal": []},
    }

    # ── PHASE 1: all models, all seeds, all fractions ─────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1 — TRAINING ALL MODELS")
    print("=" * 70)

    for seed in SEEDS:
        print(f"\n{'─'*60}")
        print(f"  SEED {seed}")
        print(f"{'─'*60}")

        for fraction in FRACTIONS:
            print(f"\n  Fraction {fraction*100:.0f}%")

            # CNN+LSTM
            print("    [CNN+LSTM] multimodal")
            all_results["cnn_lstm"]["multimodal"].append(
                train_cnn_lstm(CNNLSTM, seed, fraction,
                               train_df, val_df, test_df, mbert_tok, "multimodal"))
            print("    [CNN+LSTM] text-only")
            all_results["cnn_lstm"]["text_only"].append(
                train_cnn_lstm(CNNLSTMTextOnly, seed, fraction,
                               train_df, val_df, test_df, mbert_tok, "text_only"))
            print("    [CNN+LSTM] image-only")
            all_results["cnn_lstm"]["image_only"].append(
                train_cnn_lstm(CNNLSTMImageOnly, seed, fraction,
                               train_df, val_df, test_df, mbert_tok, "image_only"))

            # ViT+TCN
            print("    [ViT+TCN] multimodal")
            all_results["vit_tcn"]["multimodal"].append(
                train_vit_tcn(ViTTCNModel, vit_tcn_base, seed, fraction,
                              train_df, val_df, test_df, mbert_tok, "multimodal"))
            print("    [ViT+TCN] text-only")
            all_results["vit_tcn"]["text_only"].append(
                train_vit_tcn(ViTTCNTextOnly, None, seed, fraction,
                              train_df, val_df, test_df, mbert_tok, "text_only"))
            print("    [ViT+TCN] image-only")
            all_results["vit_tcn"]["image_only"].append(
                train_vit_tcn(ViTTCNImageOnly, vit_tcn_base, seed, fraction,
                              train_df, val_df, test_df, mbert_tok, "image_only"))

            # ResNet-50 + mBERT
            print("    [ResNet+mBERT] multimodal")
            all_results["resnet_mbert"]["multimodal"].append(
                train_resnet_mbert(ResNetMBERT, seed, fraction,
                                   train_df, val_df, test_df, mbert_tok, "multimodal"))
            print("    [ResNet+mBERT] text-only")
            all_results["resnet_mbert"]["text_only"].append(
                train_resnet_mbert(ResNetMBERTTextOnly, seed, fraction,
                                   train_df, val_df, test_df, mbert_tok, "text_only"))
            print("    [ResNet+mBERT] image-only")
            all_results["resnet_mbert"]["image_only"].append(
                train_resnet_mbert(ResNetMBERTImageOnly, seed, fraction,
                                   train_df, val_df, test_df, mbert_tok, "image_only"))

            # CLIP
            print("    [CLIP] multimodal")
            all_results["clip"]["multimodal"].append(
                train_clip_model(CLIPMultimodal, seed, fraction,
                                 train_df, val_df, test_df, clip_preprocess, "multimodal"))
            print("    [CLIP] text-only")
            all_results["clip"]["text_only"].append(
                train_clip_model(CLIPTextOnly, seed, fraction,
                                 train_df, val_df, test_df, clip_preprocess, "text_only"))
            print("    [CLIP] image-only")
            all_results["clip"]["image_only"].append(
                train_clip_model(CLIPImageOnly, seed, fraction,
                                 train_df, val_df, test_df, clip_preprocess, "image_only"))

            # ViT+MuRIL
            print("    [ViT+MuRIL] multimodal")
            all_results["vit_muril"]["multimodal"].append(
                train_vit_muril("multimodal", vit_muril_base, muril_base,
                                seed, fraction, train_df, val_df, test_df, muril_tok))
            print("    [ViT+MuRIL] text-only")
            all_results["vit_muril"]["text_only"].append(
                train_vit_muril("text_only", vit_muril_base, muril_base,
                                seed, fraction, train_df, val_df, test_df, muril_tok))
            print("    [ViT+MuRIL] image-only")
            all_results["vit_muril"]["image_only"].append(
                train_vit_muril("image_only", vit_muril_base, muril_base,
                                seed, fraction, train_df, val_df, test_df, muril_tok))

            # Standalone text-only baselines (Table VI dedicated rows)
            print("    [text-only mBERT]")
            all_results["standalone_mbert"]["multimodal"].append(
                train_standalone_mbert(seed, fraction,
                                       train_df, val_df, test_df, mbert_tok))
            print("    [text-only MuRIL]")
            all_results["standalone_muril"]["multimodal"].append(
                train_standalone_muril(muril_base, seed, fraction,
                                       train_df, val_df, test_df, muril_tok))

    # ── PHASE 2: Print paper tables ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2 — PAPER TABLES")
    print("=" * 70)

    print("\n📊 TABLE VI — Main benchmark results (100% data, 5 seeds, ddof=1)")
    print(f"  {'Model':<25s}  {'Acc%':>10s}  {'F1%':>10s}  {'AUC':>12s}")
    for name, key in [
        ("text-only mBERT",    ("standalone_mbert", "multimodal")),
        ("text-only MuRIL",    ("standalone_muril", "multimodal")),
        ("ResNet-50+mBERT",    ("resnet_mbert",     "multimodal")),
        ("ViT+MuRIL",          ("vit_muril",        "multimodal")),
        ("ViT+TCN",            ("vit_tcn",          "multimodal")),
        ("CNN+LSTM",           ("cnn_lstm",         "multimodal")),
        ("CLIP",               ("clip",             "multimodal")),
    ]:
        print_table_vi(name, all_results[key[0]][key[1]])

    print("\n📊 TABLE VIII — Scaling (Macro-F1, all fractions)")
    for name, key in [
        ("ResNet-50+mBERT", ("resnet_mbert", "multimodal")),
        ("ViT+MuRIL",       ("vit_muril",    "multimodal")),
        ("ViT+TCN",         ("vit_tcn",      "multimodal")),
        ("CNN+LSTM",        ("cnn_lstm",     "multimodal")),
        ("CLIP",            ("clip",         "multimodal")),
    ]:
        row = f"  {name:<20s}  "
        for frac in FRACTIONS:
            s = compute_stats(all_results[key[0]][key[1]], fraction=frac)
            row += f"  {frac*100:.0f}%: {s['f1_mean']*100:.2f}±{s['f1_std']*100:.2f}" if s else "  —"
        print(row)

    print("\n📊 TABLE XI — Modality ablation (100% data)")
    for name, key in [
        ("CNN+LSTM",        "cnn_lstm"),
        ("ViT+TCN",         "vit_tcn"),
        ("ResNet-50+mBERT", "resnet_mbert"),
        ("ViT+MuRIL",       "vit_muril"),
        ("CLIP",            "clip"),
    ]:
        mm = compute_stats(all_results[key]["multimodal"], fraction=1.0)
        to = compute_stats(all_results[key]["text_only"],  fraction=1.0)
        io = compute_stats(all_results[key]["image_only"], fraction=1.0)
        if mm and to and io:
            print(f"  {name:<20s}  MM={mm['f1_mean']*100:.2f}  "
                  f"Text={to['f1_mean']*100:.2f}  "
                  f"Img={io['f1_mean']*100:.2f}")

    print("\n📊 TABLE XII — Per-typology OOC F1 (100% data)")
    for name, key in [
        ("ResNet-50+mBERT", ("resnet_mbert", "multimodal")),
        ("ViT+MuRIL",       ("vit_muril",    "multimodal")),
        ("ViT+TCN",         ("vit_tcn",      "multimodal")),
        ("CNN+LSTM",        ("cnn_lstm",     "multimodal")),
        ("CLIP",            ("clip",         "multimodal")),
    ]:
        full_res = [r for r in all_results[key[0]][key[1]] if r["fraction"] == 1.0]
        typo_stats = per_typology_f1_across_seeds(full_res)
        print(f"\n  {name}")
        for typo in TYPO_KEYS:
            s = typo_stats.get(typo)
            if s:
                print(f"    {TYPO_DISPLAY[typo]:<22s}  "
                      f"F1={s['mean']*100:.2f}±{s['std']*100:.2f}  (n_seeds={s['n']})")

    # ── PHASE 3: McNemar's tests ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 3 — McNEMAR'S TESTS")
    print("=" * 70)

    p1 = run_mcnemar(
        all_results["standalone_mbert"]["multimodal"],
        all_results["resnet_mbert"]["multimodal"],
        "text-only mBERT", "ResNet-50+mBERT")

    p2 = run_mcnemar(
        all_results["vit_muril"]["multimodal"],
        all_results["resnet_mbert"]["multimodal"],
        "ViT+MuRIL", "ResNet-50+mBERT")

    p3 = run_mcnemar(
        all_results["standalone_muril"]["multimodal"],
        all_results["standalone_mbert"]["multimodal"],
        "text-only MuRIL", "text-only mBERT")

    mcnemar_results = {
        "mBERT_vs_ResNetmBERT":   p1.tolist(),
        "ViTMuRIL_vs_ResNetmBERT": p2.tolist(),
        "MuRIL_vs_mBERT":         p3.tolist(),
    }

    # ── PHASE 4: Leakage validation ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 4 — LEAKAGE VALIDATION (event-cluster split)")
    print("=" * 70)

    cluster_ids = build_event_clusters(full_df)
    cluster_train, cluster_test = make_cluster_split(full_df, cluster_ids)
    # Use 10% of cluster_train as validation
    val_size       = max(1, int(0.10 * len(cluster_train)))
    cluster_val    = cluster_train.iloc[:val_size].reset_index(drop=True)
    cluster_train2 = cluster_train.iloc[val_size:].reset_index(drop=True)

    leakage_results = {}
    leakage_deltas  = {}

    for model_label, run_fn in [
        ("text_mbert",   lambda s: train_standalone_mbert(
            s, 1.0, cluster_train2, cluster_val, cluster_test, mbert_tok)),
        ("text_muril",   lambda s: train_standalone_muril(
            muril_base, s, 1.0, cluster_train2, cluster_val, cluster_test, muril_tok)),
        ("resnet_mbert", lambda s: train_resnet_mbert(
            ResNetMBERT, s, 1.0, cluster_train2, cluster_val, cluster_test, mbert_tok)),
        ("vit_muril",    lambda s: train_vit_muril(
            "multimodal", vit_muril_base, muril_base,
            s, 1.0, cluster_train2, cluster_val, cluster_test, muril_tok)),
        ("vit_tcn",      lambda s: train_vit_tcn(
            ViTTCNModel, vit_tcn_base,
            s, 1.0, cluster_train2, cluster_val, cluster_test, mbert_tok)),
        ("cnn_lstm",     lambda s: train_cnn_lstm(
            CNNLSTM, s, 1.0, cluster_train2, cluster_val, cluster_test, mbert_tok)),
        ("clip",         lambda s: train_clip_model(
            CLIPMultimodal, s, 1.0,
            cluster_train2, cluster_val, cluster_test, clip_preprocess)),
    ]:
        print(f"\n  Cluster split → {model_label}")
        clust_res = []
        for seed in SEEDS:
            r = run_fn(seed)
            r["fraction"] = 1.0
            clust_res.append(r)
        leakage_results[model_label] = clust_res

        # Compare with random split accuracy
        rand_key  = model_label if model_label not in ("text_mbert", "text_muril") \
                    else ("standalone_mbert" if model_label == "text_mbert" else "standalone_muril")
        rand_mode = "multimodal"
        rand_res  = [r for r in all_results[rand_key][rand_mode] if r["fraction"] == 1.0]
        rand_acc  = np.mean([r["acc"] for r in rand_res])
        clust_acc = np.mean([r["acc"] for r in clust_res])
        delta     = clust_acc - rand_acc
        leakage_deltas[model_label] = float(delta)

    print("\n📊 LEAKAGE VALIDATION TABLE")
    print(f"  {'Model':<20s}  {'Random acc':>12s}  {'Cluster acc':>12s}  {'Δ':>8s}")
    for mdl, delta in leakage_deltas.items():
        rand_key  = mdl if mdl not in ("text_mbert", "text_muril") \
                    else ("standalone_mbert" if mdl == "text_mbert" else "standalone_muril")
        rand_res  = [r for r in all_results[rand_key]["multimodal"] if r["fraction"] == 1.0]
        rand_acc  = np.mean([r["acc"] for r in rand_res])
        clust_acc = rand_acc + delta
        leak_flag = "← LEAKAGE" if abs(delta) > 0.02 else "✓ clean"
        print(f"  {mdl:<20s}  {rand_acc*100:>10.2f}%  "
              f"{clust_acc*100:>10.2f}%  {delta*100:>+7.2f}%  {leak_flag}")

    # ── PHASE 5: Save everything ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 5 — SAVING RESULTS")
    print("=" * 70)

    def serialise(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)): return obj.item()
        return obj

    def save_results(results_list: list, fname: str):
        rows = []
        for r in results_list:
            rows.append({k: serialise(v) for k, v in r.items()
                         if not isinstance(v, (list, np.ndarray))
                         or k in ("confusion_matrix",)})
        pd.DataFrame(rows).to_csv(OUT_DIR / fname, index=False)

    for model_name, modes in all_results.items():
        for mode, res in modes.items():
            if res:
                save_results(res, f"{model_name}_{mode}.csv")

    with open(OUT_DIR / "mcnemar_pvalues.json", "w") as f:
        json.dump(mcnemar_results, f, indent=2)

    leakage_summary = []
    for mdl, delta in leakage_deltas.items():
        rand_key = mdl if mdl not in ("text_mbert", "text_muril") \
                   else ("standalone_mbert" if mdl == "text_mbert" else "standalone_muril")
        rand_res = [r for r in all_results[rand_key]["multimodal"] if r["fraction"] == 1.0]
        rand_acc = float(np.mean([r["acc"] for r in rand_res]))
        leakage_summary.append({
            "model":             mdl,
            "random_split_acc":  rand_acc,
            "cluster_split_acc": rand_acc + delta,
            "delta":             delta,
        })
    pd.DataFrame(leakage_summary).to_csv(OUT_DIR / "leakage_validation.csv", index=False)

    print(f"\n✅ All results saved to {OUT_DIR}")
    print(f"   Files written:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"   • {f.name}")


if __name__ == "__main__":
    main()