"""
nepooc_vit_muril_single_seed.py  |  v3 — TEXT->IMAGE CROSS-ATTENTION (FULLY FIXED)
ViT+MuRIL — Dual-Stream Region-Aware Architecture (Model E)

FIXES INCLUDED:
  1. ViT patch_embed.img_size, grid_size, num_patches updated for 448×448 input.
  2. Cross‑attention direction: text CLS as query, image patches as K/V.
  3. Stratified subsampling for all data fractions (preserves 50/50 balance).
  4. Robust image path resolution with fallback.
  5. NaN label and caption handling.
  6. Gradient clipping, label smoothing, adaptive LR (paper Table VI).
  7. ROC/PR curves and per‑seed JSON output.
"""

import os
import sys
import copy
import random
import time
import json
import warnings
from typing import Optional
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import timm
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# ============================================================
# SEED CONFIGURATION
# ============================================================
_env_seed = os.environ.get("NEPOOC_SEED", None)
SEED = int(_env_seed) if _env_seed is not None else 42  # ← CHANGE PER NOTEBOOK

ALL_SEEDS = [42, 123, 456, 789, 2024]
assert SEED in ALL_SEEDS, f"SEED={SEED} not in {ALL_SEEDS}"
print(f"{'='*70}")
print(f"  Running single-seed experiment:  SEED = {SEED}")
print(f"  Cross‑attention: TEXT → IMAGE (text CLS queries image patches)")
print(f"{'='*70}")

# ============================================================
# HYPERPARAMETERS (Table VI)
# ============================================================
FRACTIONS    = [0.25, 0.50, 0.75, 1.0]
BATCH_SIZE   = 8
EPOCHS       = 100
WEIGHT_DECAY = 0.05
PATIENCE     = 12
NUM_CLASSES  = 2
NUM_WORKERS  = 4
MAX_TEXT_LEN = 128
IMG_SIZE     = 448
LORA_RANK    = 8
LORA_ALPHA   = 16
LABEL_SMOOTH = 0.1
WARMUP_FRAC  = 0.10
GRAD_CLIP    = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# PATHS — update to match your Kaggle dataset slugs
# ============================================================
CSV_DIR    = Path("/kaggle/input/datasets/amanlamichhane1234/nepooc-datset")
IMG_DIR    = Path("/kaggle/input/datasets/sanjeevkhatiwada/nepali-ooc-images")
OUTPUT_DIR = Path(f"/kaggle/working/vit_muril_seed{SEED}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MURIL_MODEL_NAME = "google/muril-base-cased"

TYPO_DISPLAY = {
    "Fabricated":          "Fabricated",
    "Miscaptioned":        "Miscaptioned",
    "Temporal_Mismatch":   "Temporal mismatch",
    "Geographic_Mismatch": "Geographic mismatch",
    "Identity_Mismatch":   "Identity mismatch",
    "pristine":            "Pristine",
}

# ============================================================
# EMPTY DATALOADER GUARD
# ============================================================
def assert_nonempty_split(df: pd.DataFrame, name: str) -> None:
    if len(df) == 0:
        raise RuntimeError(
            f"[FATAL] Split '{name}' is empty after image-path resolution. "
            f"Check that IMG_DIR='{IMG_DIR}' contains the expected files and "
            f"that CSV_DIR='{CSV_DIR}' has the correct CSVs."
        )
    classes = df["label"].unique()
    if len(classes) < 2:
        raise RuntimeError(
            f"[FATAL] Split '{name}' has only class(es) {classes.tolist()} "
            f"after filtering. Stratification will fail."
        )

# ============================================================
# NaN-LABEL GUARD
# ============================================================
def drop_nan_labels(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    n_before = len(df)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  [WARNING] Dropped {n_dropped} rows with NaN labels from '{split_name}'.")
    return df

# ============================================================
# LoRA LAYER
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = linear
        self.d_in   = linear.weight.shape[1]
        self.d_out  = linear.weight.shape[0]

        for p in self.linear.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(rank, self.d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.d_out, rank))
        self.scale  = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_in, (
            f"LoRALinear expected last dim={self.d_in}, got {x.shape[-1]}"
        )
        return self.linear(x) + self.scale * (x @ self.lora_A.T @ self.lora_B.T)

def apply_lora_to_muril(model: nn.Module, rank: int, alpha: int) -> None:
    for layer in model.encoder.layer:
        attn_self = layer.attention.self
        attn_self.query = LoRALinear(attn_self.query, rank=rank, alpha=alpha)
        attn_self.value = LoRALinear(attn_self.value, rank=rank, alpha=alpha)

    for name, param in model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name:
            param.requires_grad = False

# ============================================================
# ViT-B/16 @ 448×448 WITH BICUBIC POSITIONAL INTERPOLATION (FIXED)
# ============================================================
def build_vit_448(pretrained: bool = True) -> nn.Module:
    vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained)

    # --- FIX: Update patch embedding to accept 448×448 input ---
    new_img_size = (IMG_SIZE, IMG_SIZE)          # (448,448)
    vit.patch_embed.img_size = new_img_size
    grid_size = (IMG_SIZE // 16, IMG_SIZE // 16) # (28,28)
    vit.patch_embed.grid_size = grid_size
    vit.patch_embed.num_patches = grid_size[0] * grid_size[1]  # 784

    with torch.no_grad():
        pos_embed   = vit.pos_embed          # (1, 197, 768)
        cls_token   = pos_embed[:, :1, :]
        patch_embed = pos_embed[:, 1:, :]

        old_grid = int(patch_embed.shape[1] ** 0.5)   # 14
        new_grid = IMG_SIZE // 16                      # 28

        patch_embed = (
            patch_embed
            .reshape(1, old_grid, old_grid, 768)
            .permute(0, 3, 1, 2)
            .float()
        )
        patch_embed = F.interpolate(
            patch_embed,
            size=(new_grid, new_grid),
            mode="bicubic",
            align_corners=False,
        )
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, 768)
        new_pos_embed = torch.cat([cls_token, patch_embed], dim=1)   # (1, 785, 768)
        vit.pos_embed = nn.Parameter(new_pos_embed)

    for param in vit.parameters():
        param.requires_grad = False

    return vit

# ============================================================
# CROSS-ATTENTION: TEXT → IMAGE (single text query)
# ============================================================
class TextToImageCrossAttention(nn.Module):
    def __init__(self, d_model: int = 768, dk: int = 64):
        super().__init__()
        self.dk    = dk
        self.W_Q   = nn.Linear(d_model, dk, bias=False)
        self.W_K   = nn.Linear(d_model, dk, bias=False)
        self.W_V   = nn.Linear(d_model, dk, bias=False)
        self.W_out = nn.Linear(dk, d_model, bias=False)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, text_query: torch.Tensor,   # (B, d_model)
                v_patches: torch.Tensor) -> torch.Tensor:  # (B, num_patches, d_model)
        Q = self.W_Q(text_query).unsqueeze(1)          # (B, 1, dk)
        K = self.W_K(v_patches)                        # (B, num_patches, dk)
        V = self.W_V(v_patches)                        # (B, num_patches, dk)

        scores = torch.matmul(Q, K.transpose(1, 2)) / (self.dk ** 0.5)  # (B, 1, num_patches)
        attn   = torch.softmax(scores, dim=-1)                            # (B, 1, num_patches)
        out    = torch.matmul(attn, V)                                     # (B, 1, dk)
        out    = self.W_out(out).squeeze(1)                                # (B, d_model)

        return self.norm(out + text_query)

# ============================================================
# MODELS
# ============================================================
class ViTMuRILMultimodal(nn.Module):
    def __init__(self, vit: nn.Module, muril: nn.Module):
        super().__init__()
        self.vit        = vit
        self.muril      = muril
        self.cross_attn = TextToImageCrossAttention(d_model=768, dk=64)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, images, input_ids, attention_mask):
        vit_out   = self.vit.forward_features(images)          # (B, 785, 768)
        v_patches = vit_out[:, 1:, :]                          # (B, 784, 768)

        muril_out = self.muril(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state                                    # (B, 128, 768)
        text_cls  = muril_out[:, 0, :]                         # (B, 768)

        fused = self.cross_attn(text_cls, v_patches)           # (B, 768)
        return self.classifier(fused)

class ViTMuRILTextOnly(nn.Module):
    def __init__(self, muril: nn.Module):
        super().__init__()
        self.muril      = muril
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, images, input_ids, attention_mask):
        cls = self.muril(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        return self.classifier(cls)

class ViTMuRILImageOnly(nn.Module):
    def __init__(self, vit: nn.Module):
        super().__init__()
        self.vit        = vit
        self.classifier = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, images, input_ids, attention_mask):
        cls = self.vit.forward_features(images)[:, 0, :]
        return self.classifier(cls)

# ============================================================
# IMAGE TRANSFORM
# ============================================================
vit_transform = T.Compose([
    T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================
# DATASET
# ============================================================
class NepOOCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, transform=vit_transform):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            img = self.transform(img)
        except Exception:
            print(f"  [WARNING] Corrupt/missing image: {row['image_path']} — using blank.")
            img = self.transform(Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0)))

        # Caption: NaN → empty string
        caption = str(row["caption"]).strip() if pd.notna(row["caption"]) else ""

        enc = self.tokenizer(
            caption,
            max_length=MAX_TEXT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        raw_type = row.get("misinformation_type", None)
        mtype    = "pristine" if (raw_type is None or pd.isna(raw_type)) else str(raw_type)

        return {
            "image":               img,
            "input_ids":           enc["input_ids"].squeeze(0),
            "attention_mask":      enc["attention_mask"].squeeze(0),
            "label":               torch.tensor(int(row["label"]), dtype=torch.long),
            "post_id":             str(row["post_id"]),
            "misinformation_type": mtype,
        }

# ============================================================
# SCHEDULER
# ============================================================
def get_cosine_warmup_scheduler(optimizer, total_steps: int, warmup_frac: float = 0.10):
    warmup_steps = int(total_steps * warmup_frac)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================================
# UTILITIES
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================
# EVALUATION
# ============================================================
def evaluate(model: nn.Module, loader: DataLoader, device, context: str = "") -> dict:
    if len(loader.dataset) == 0:
        raise RuntimeError(f"evaluate() called with empty dataset. Context: {context}")

    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    all_post_ids, all_types          = [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["label"]

            logits = model(images, ids, mask)
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu()
            preds  = logits.argmax(dim=1).cpu()

            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_post_ids.extend(batch["post_id"])
            all_types.extend(batch["misinformation_type"])

    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    prec_ooc = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
    rec_ooc  = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
    f1_ooc   = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    cm       = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError as e:
        print(f"  [WARNING] AUC computation failed ({context}): {e}. Setting AUC=0.0.")
        auc = 0.0

    return {
        "acc":           acc,
        "f1_macro":      f1,
        "auc":           auc,
        "precision_ooc": prec_ooc,
        "recall_ooc":    rec_ooc,
        "f1_ooc":        f1_ooc,
        "cm":            cm.tolist(),
        "labels":        all_labels,
        "preds":         all_preds,
        "probs":         all_probs,
        "post_ids":      all_post_ids,
        "types":         all_types,
    }

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        loss = criterion(model(images, ids, mask), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=GRAD_CLIP,
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(1, n_batches)

def build_model(model_type: str, vit_base, muril_base, device):
    muril = copy.deepcopy(muril_base)
    vit   = copy.deepcopy(vit_base)

    if model_type in ("multimodal", "text_only"):
        apply_lora_to_muril(muril, rank=LORA_RANK, alpha=LORA_ALPHA)

    if model_type == "multimodal":
        m = ViTMuRILMultimodal(vit, muril)
    elif model_type == "text_only":
        m = ViTMuRILTextOnly(muril)
    else:  # image_only
        m = ViTMuRILImageOnly(vit)

    return m.to(device)

# ============================================================
# SINGLE RUN
# ============================================================
def train_model(
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    test_df:    pd.DataFrame,
    tokenizer,
    vit_base,
    muril_base,
    seed:       int,
    fraction:   float,
    model_type: str,
) -> dict:
    set_seed(seed)

    if fraction < 1.0:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=1.0 - fraction, random_state=seed
        )
        keep_idx, _ = next(sss.split(train_df, train_df["label"]))
        sub_train   = train_df.iloc[keep_idx].reset_index(drop=True)
    else:
        sub_train = train_df.reset_index(drop=True)

    assert_nonempty_split(sub_train, f"sub_train(frac={fraction})")

    lr = 1e-4 if fraction <= 0.50 else 5e-5

    train_ds = NepOOCDataset(sub_train, tokenizer)
    val_ds   = NepOOCDataset(val_df,    tokenizer)
    test_ds  = NepOOCDataset(test_df,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = build_model(model_type, vit_base, muril_base, DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = get_cosine_warmup_scheduler(
        optimizer, len(train_loader) * EPOCHS, WARMUP_FRAC
    )

    best_val_f1    = -1.0
    patience_count = 0
    best_state     = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE)
        val_res    = evaluate(model, val_loader, DEVICE,
                              context=f"val seed={seed} frac={fraction} {model_type} ep={epoch}")
        val_f1 = val_res["f1_macro"]

        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"    [Early stop @ epoch {epoch}]  best val F1={best_val_f1:.4f}")
            break

    if best_state is None:
        print("  [WARNING] best_state is None — training loop did not execute. "
              "Saving current weights as fallback.")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    test_res = evaluate(model, test_loader, DEVICE,
                        context=f"test seed={seed} frac={fraction} {model_type}")

    test_res.update({
        "seed":         seed,
        "fraction":     fraction,
        "best_val_f1":  best_val_f1,
        "model_type":   model_type,
        "n_trainable":  count_trainable_params(model),
    })

    del model
    torch.cuda.empty_cache()
    return test_res

# ============================================================
# STATISTICS (single seed)
# ============================================================
def compute_stats(results: list, fraction: float = 1.0) -> Optional[dict]:
    subset = [r for r in results if r["fraction"] == fraction]
    if not subset:
        return None

    accs  = [r["acc"]           for r in subset]
    f1s   = [r["f1_macro"]      for r in subset]
    aucs  = [r["auc"]           for r in subset]
    precs = [r["precision_ooc"] for r in subset]
    recs  = [r["recall_ooc"]    for r in subset]
    cms   = np.array([r["cm"]   for r in subset])

    return {
        "acc_mean":  np.mean(accs),  "acc_std":  np.std(accs),
        "f1_mean":   np.mean(f1s),   "f1_std":   np.std(f1s),
        "auc_mean":  np.mean(aucs),  "auc_std":  np.std(aucs),
        "prec_mean": np.mean(precs), "prec_std": np.std(precs),
        "rec_mean":  np.mean(recs),  "rec_std":  np.std(recs),
        "confusion_matrix": cms.mean(axis=0).tolist(),
        "n_seeds":   len(subset),
    }

# ============================================================
# LOAD DATA
# ============================================================
print(f"\n{'='*70}")
print("LOADING DATASET")
print(f"{'='*70}")
print("NOTE: Using full 1,090-sample splits (train≈754/val≈108/test≈228).")


train_df = pd.read_csv(CSV_DIR / "nepOOC_train.csv")
val_df   = pd.read_csv(CSV_DIR / "nepOOC_val.csv")
test_df  = pd.read_csv(CSV_DIR / "nepOOC_test.csv")

train_df = drop_nan_labels(train_df, "train")
val_df   = drop_nan_labels(val_df,   "val")
test_df  = drop_nan_labels(test_df,  "test")

original_counts = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}

# Resolve image paths
def get_image_path(post_id: str, img_dir: Path) -> Optional[str]:
    # Try exact match first
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        path = img_dir / f"{post_id}{ext}"
        if path.exists():
            return str(path)
    # Fallback to recursive glob
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        matches = list(img_dir.rglob(f"*{post_id}*{ext}"))
        if matches:
            return str(matches[0])
    return None

for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    df["image_path"] = df["post_id"].apply(lambda pid: get_image_path(pid, IMG_DIR))

train_df = train_df[train_df["image_path"].notna()].reset_index(drop=True)
val_df   = val_df[val_df["image_path"].notna()].reset_index(drop=True)
test_df  = test_df[test_df["image_path"].notna()].reset_index(drop=True)

assert_nonempty_split(train_df, "train (after image resolution)")
assert_nonempty_split(val_df,   "val   (after image resolution)")
assert_nonempty_split(test_df,  "test  (after image resolution)")

print("✅ Dataset loaded:")
for split, df, orig in [
    ("Train", train_df, original_counts["train"]),
    ("Val",   val_df,   original_counts["val"]),
    ("Test",  test_df,  original_counts["test"]),
]:
    ooc  = int(df["label"].sum())
    pris = len(df) - ooc
    print(f"   {split}: {len(df)}/{orig}  (OOC={ooc}, Pristine={pris})")

# ============================================================
# LOAD BASE MODELS
# ============================================================
print(f"\n{'='*70}")
print("LOADING BASE MODELS")
print(f"{'='*70}")

print("📦 ViT-B/16 @ 448×448 ...")
vit_base = build_vit_448(pretrained=True)
print(f"   pos_embed shape: {vit_base.pos_embed.shape}")   # (1, 785, 768)

print("📦 MuRIL tokenizer + model ...")
tokenizer  = AutoTokenizer.from_pretrained(MURIL_MODEL_NAME)
muril_base = AutoModel.from_pretrained(MURIL_MODEL_NAME)
for param in muril_base.parameters():
    param.requires_grad = False

print(f"   MuRIL vocab size : {tokenizer.vocab_size:,}")
print(f"   MuRIL parameters : {sum(p.numel() for p in muril_base.parameters()):,}")

_tmp = build_model("multimodal", vit_base, muril_base, "cpu")
print(f"\n✅ Trainable params (multimodal): {count_trainable_params(_tmp):,}  (≈2.5M expected)")
del _tmp

# ============================================================
# MAIN EXPERIMENT LOOP — single seed, all fractions, all model types
# ============================================================
print(f"\n{'='*70}")
print(f"EXPERIMENTS  seed={SEED}  |  4 fractions × 3 model types = 12 runs")
print(f"Cross‑attention: TEXT → IMAGE (text CLS queries image patches)")
print(f"{'='*70}")

all_results   = {"multimodal": [], "text_only": [], "image_only": []}
typo_per_type = {
    "multimodal": defaultdict(lambda: {"labels": [], "preds": []}),
    "text_only":  defaultdict(lambda: {"labels": [], "preds": []}),
    "image_only": defaultdict(lambda: {"labels": [], "preds": []}),
}

total_runs = len(FRACTIONS) * 3
run_count  = 0
t0_total   = time.time()

for frac in FRACTIONS:
    for mtype in ("multimodal", "text_only", "image_only"):
        run_count += 1
        lr_str = "1e-4" if frac <= 0.5 else "5e-5"
        print(f"\n[{run_count}/{total_runs}]  seed={SEED}  frac={frac:.0%}  "
              f"type={mtype}  lr={lr_str}")
        t0 = time.time()

        result = train_model(
            train_df, val_df, test_df,
            tokenizer, vit_base, muril_base,
            seed=SEED, fraction=frac, model_type=mtype,
        )
        elapsed = time.time() - t0
        print(f"   → acc={result['acc']:.3f}  F1={result['f1_macro']:.3f}  "
              f"AUC={result['auc']:.3f}  val_F1={result['best_val_f1']:.3f}  "
              f"({elapsed:.0f}s)")

        all_results[mtype].append(result)

        # Collect typology data at 100% data for all three model types
        if frac == 1.0:
            for pred, label, mtyp in zip(
                result["preds"], result["labels"], result["types"]
            ):
                if label == 1:  # OOC only
                    typo_per_type[mtype][mtyp]["labels"].append(label)
                    typo_per_type[mtype][mtyp]["preds"].append(pred)

elapsed_total = time.time() - t0_total
print(f"\n✅ All 12 runs complete in {elapsed_total/60:.1f} min")

# ============================================================
# SINGLE-SEED TABLE PRINTOUT
# ============================================================
print(f"\n{'='*70}")
print(f"SINGLE-SEED RESULTS  (seed={SEED}, fraction=100%)")
print(f"{'='*70}")

for mtype_key in ("multimodal", "text_only", "image_only"):
    s = compute_stats(all_results[mtype_key], fraction=1.0)
    if s is None:
        print(f"\n  [{mtype_key}] No results at fraction=1.0")
        continue
    res = [r for r in all_results[mtype_key] if r["fraction"] == 1.0][0]
    print(f"\n  [{mtype_key}]")
    print(f"    Acc      : {res['acc']*100:.1f}%")
    print(f"    Macro-F1 : {res['f1_macro']:.3f}")
    print(f"    AUC      : {res['auc']:.3f}")
    print(f"    OOC Prec : {res['precision_ooc']:.3f}")
    print(f"    OOC Rec  : {res['recall_ooc']:.3f}")

print(f"\n  Scaling (multimodal):")
for frac in FRACTIONS:
    s = compute_stats(all_results["multimodal"], fraction=frac)
    if s is not None:
        res = [r for r in all_results["multimodal"] if r["fraction"] == frac][0]
        print(f"    {frac*100:3.0f}%  F1={res['f1_macro']:.3f}")
    else:
        print(f"    {frac*100:3.0f}%  (no result)")

print(f"\n  Typology (multimodal, 100% data, seed={SEED}):")
all_typologies = set(typo_per_type["multimodal"].keys())
for typo in sorted(all_typologies):
    data = typo_per_type["multimodal"].get(typo)
    if data and data["labels"]:
        f1 = f1_score(data["labels"], data["preds"], pos_label=1, zero_division=0)
        display = TYPO_DISPLAY.get(typo, typo)
        print(f"    {display:20s} : F1={f1:.3f}  (n={len(data['labels'])})")
    else:
        display = TYPO_DISPLAY.get(typo, typo)
        print(f"    {display:20s} : no OOC samples this seed")

# ============================================================
# SAVE SEED-SPECIFIC RESULTS
# ============================================================
print(f"\n💾 Saving results to {OUTPUT_DIR}")

for mtype_key in ("multimodal", "text_only", "image_only"):
    rows = []
    for r in all_results[mtype_key]:
        rows.append({
            "seed":         r["seed"],
            "fraction":     r["fraction"],
            "test_acc":     r["acc"],
            "test_f1":      r["f1_macro"],
            "test_auc":     r["auc"],
            "ooc_prec":     r["precision_ooc"],
            "ooc_rec":      r["recall_ooc"],
            "ooc_f1":       r["f1_ooc"],
            "best_val_f1":  r["best_val_f1"],
            "n_trainable":  r["n_trainable"],
            "cm":           str(r["cm"]),
        })
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / f"seed{SEED}_{mtype_key}_results.csv", index=False
    )

# Typology CSV (multimodal)
typo_rows = []
for typo, data in typo_per_type["multimodal"].items():
    if data["labels"]:
        typo_rows.append({
            "typology": typo,
            "seed":     SEED,
            "f1":       f1_score(data["labels"], data["preds"],
                                 pos_label=1, zero_division=0),
            "n_samples": len(data["labels"]),
        })
pd.DataFrame(typo_rows).to_csv(
    OUTPUT_DIR / f"seed{SEED}_typology_results.csv", index=False
)

# Full JSON for aggregation script
save_payload = {}
for mtype_key, results_list in all_results.items():
    save_payload[mtype_key] = []
    for r in results_list:
        entry = {k: v for k, v in r.items()}
        for k, v in entry.items():
            if isinstance(v, np.integer):
                entry[k] = int(v)
            elif isinstance(v, np.floating):
                entry[k] = float(v)
            elif isinstance(v, np.ndarray):
                entry[k] = v.tolist()
        save_payload[mtype_key].append(entry)

with open(OUTPUT_DIR / f"seed{SEED}_all_results_full.json", "w") as f:
    json.dump(save_payload, f, indent=2)

print(f"✅ Saved:")
print(f"   seed{SEED}_multimodal_results.csv")
print(f"   seed{SEED}_text_only_results.csv")
print(f"   seed{SEED}_image_only_results.csv")
print(f"   seed{SEED}_typology_results.csv")
print(f"   seed{SEED}_all_results_full.json  ← use this for aggregation")

# ============================================================
# ROC + PR CURVES for this seed at fraction=1.0 (multimodal)
# ============================================================
mm_results_full = [r for r in all_results["multimodal"] if r["fraction"] == 1.0]
if mm_results_full:
    r = mm_results_full[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    try:
        fpr, tpr, _ = roc_curve(r["labels"], r["probs"])
        ax1.plot(fpr, tpr, linewidth=2, color="steelblue",
                 label=f'ViT+MuRIL (AUC={r["auc"]:.3f})')
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
        ax1.set_title(f"ROC (seed={SEED})")
        ax1.legend(); ax1.grid(True, alpha=0.3)
    except Exception as e:
        ax1.set_title(f"ROC unavailable: {e}")

    try:
        prec_vals, rec_vals, _ = precision_recall_curve(r["labels"], r["probs"])
        ax2.plot(rec_vals, prec_vals, linewidth=2, color="darkgreen")
        ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.set_title(f"PR Curve (seed={SEED})"); ax2.grid(True, alpha=0.3)
    except Exception as e:
        ax2.set_title(f"PR unavailable: {e}")

    plt.tight_layout()
    fig_path = OUTPUT_DIR / f"seed{SEED}_roc_pr.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"   seed{SEED}_roc_pr.png")

print(f"\n{'='*70}")
print(f"✅  SEED {SEED} COMPLETE — cross‐attention direction: TEXT → IMAGE")
print(f"    Share seed{SEED}_all_results_full.json with aggregation notebook.")
print(f"{'='*70}")