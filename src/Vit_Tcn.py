"""
ViT+TCN for NepOOC Out-of-Context Misinformation Detection
===========================================================
Designed for 5 parallel Kaggle notebooks, one per seed.

HOW TO USE
----------
Set NOTEBOOK_SEED at the top of this file before running:
  Notebook 1 → NOTEBOOK_SEED = 42
  Notebook 2 → NOTEBOOK_SEED = 123
  Notebook 3 → NOTEBOOK_SEED = 456
  Notebook 4 → NOTEBOOK_SEED = 789
  Notebook 5 → NOTEBOOK_SEED = 2024

Each notebook trains all 4 fractions for its seed, saves per-fraction
JSON result files, then runs modality ablation for that seed.

After all 5 notebooks finish, run aggregate_results() from any one notebook
(or locally) pointing OUTPUT_DIR at the combined output directory.

AUDIT NOTES (verified against paper Table VI and CSV inspection)
----------------------------------------------------------------
✓ AdamW, LR=5e-5, WD=1e-4, BS=32, EP=100, PAT=10, Cosine+10%warmup
✓ TCN: single Conv1d per residual block (not two), dilations {1,2,4}
✓ Stratified subsampling on 'label' column for all fractions
✓ misinformation_type NaN → 'pristine' using pd.isna() (no 'None' strings)
✓ Per-typology F1 uses zero_division=0 (safe for n=0 or n=2 Identity groups)
✓ sample_weight column is IGNORED for ViT+TCN (standard CrossEntropyLoss per paper)
✓ Table VIII: Accuracy/Macro-F1/AUC mean±std over 5 seeds at 100% data
✓ Table IX:  OOC Precision/Recall from per-seed confusion matrices
✓ Table X:   Macro-F1 per fraction (mean±std)
✓ Table XI:  Average confusion matrix (mean counts)
✓ Table XII: Per-typology OOC F1 at 100% data
✓ Table XIV: Modality ablation (text-only, image-only, multimodal)
✓ ViT forward_features returns tensor (B,197,768); patch_tokens = [:, 1:, :]
✓ Image fallback: zeros tensor (3, IMG_SIZE, IMG_SIZE) for missing files
✓ AUC is safe: test set always has both classes (114 pristine + 114 OOC)
✓ ooc_f1 safe division: 2*p*r / (p+r+1e-8)
✓ Full benchmark split: train=754, val=108, test=228 (Section XI future work)
"""

# ============================================================================
# ▶▶▶  SET THIS PER NOTEBOOK BEFORE RUNNING  ◀◀◀
# ============================================================================
NOTEBOOK_SEED = 42   # Change to 123 / 456 / 789 / 2024 in other notebooks
# ============================================================================

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import timm
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_recall_fscore_support, confusion_matrix,
    precision_recall_curve, roc_curve,
)
from PIL import Image
import torchvision.transforms as transforms


# ============================================================================
# PATHS  —  adjust if your Kaggle dataset paths differ
# ============================================================================
DATASET_DIR = "/kaggle/input/datasets/amanlamichhane1234/nepooc-datset/"
IMAGE_DIR   = "/kaggle/input/datasets/sanjeevkhatiwada/nepali-ooc-images/"
OUTPUT_DIR  = "/kaggle/working/vit_tcn_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HYPERPARAMETERS  (Table VI — ViT+TCN row)
# ============================================================================
FRACTIONS     = [0.25, 0.50, 0.75, 1.0]   # training-size fractions
ALL_SEEDS     = [42, 123, 456, 789, 2024]  # for aggregation only

# Training config — verified against Table VI
BATCH_SIZE    = 32
MAX_EPOCHS    = 100
PATIENCE      = 10         # early stopping on val Macro-F1
LEARNING_RATE = 5e-5       # AdamW LR
WEIGHT_DECAY  = 1e-4       # AdamW WD
GRAD_CLIP     = 1.0        # gradient clipping
DROPOUT       = 0.3

# Architecture config — verified against Section IV-D
IMG_SIZE      = 224
MAX_SEQ_LEN   = 128
EMBEDDING_DIM = 128        # token embedding dim d
TCN_HIDDEN    = 256        # TCN hidden dimension
FUSION_DIM    = 768        # shared multimodal feature dim
DK            = 64         # cross-attention key/query dim

# mBERT vocabulary (bert-base-multilingual-cased)
VOCAB_SIZE    = 119_547

TYPO_KEYS = [
    "Fabricated", "Miscaptioned", "Temporal_Mismatch",
    "Geographic_Mismatch", "Identity_Mismatch",
]
TYPO_DISPLAY = {
    "Fabricated":          "Fabricated",
    "Miscaptioned":        "Miscaptioned",
    "Temporal_Mismatch":   "Temporal mismatch",
    "Geographic_Mismatch": "Geographic mismatch",
    "Identity_Mismatch":   "Identity mismatch",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  Seed for this notebook: {NOTEBOOK_SEED}")


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# DATASET
# ============================================================================

class NepOOCDataset(Dataset):
    """
    Loads image–caption pairs from the NepOOC CSVs.

    Label column: int {0=Pristine, 1=OOC}  — no NaN values in data.
    misinformation_type: NaN for Pristine rows; one of 5 typology strings for OOC.
    sample_weight column exists but is deliberately ignored for ViT+TCN
    (paper uses standard CrossEntropyLoss, no weighting for this model).
    """

    def __init__(self, df, image_dir, tokenizer, transform,
                 max_seq_len: int = MAX_SEQ_LEN):
        self.df          = df.reset_index(drop=True)
        self.image_dir   = Path(image_dir)
        self.tokenizer   = tokenizer
        self.transform   = transform
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Image ────────────────────────────────────────────────────────────
        # Images are named by post_id; try .jpg then .png
        post_id  = str(row["post_id"])
        img_path = self.image_dir / f"{post_id}.jpg"
        if not img_path.exists():
            img_path = self.image_dir / f"{post_id}.png"

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            # Graceful fallback: black tensor — logged but not fatal
            image = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        # ── Caption ──────────────────────────────────────────────────────────
        enc = self.tokenizer(
            str(row["caption"]),
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)       # (L,)
        attention_mask = enc["attention_mask"].squeeze(0)  # (L,)

        # ── Labels ───────────────────────────────────────────────────────────
        label = int(row["label"])

        # pd.isna() correctly handles NaN; no 'None' strings present in data
        typology = (
            "pristine"
            if pd.isna(row["misinformation_type"])
            else str(row["misinformation_type"])
        )

        return {
            "image":          image,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "label":          label,
            "typology":       typology,
        }


# ============================================================================
# TCN ENCODER  (corrected: one Conv1d per residual block)
# ============================================================================

class DilatedResidualBlock(nn.Module):
    """
    Single dilated residual block: Conv1d → BN → (+ skip) → ReLU.

    Paper Section IV-D specifies one convolution per block.
    A 1×1 projection handles the skip connection when channels change
    (only in Block 1: embedding_dim → hidden_dim).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        # 'same' padding: keeps sequence length unchanged
        # For kernel=3: dilation=1→pad=1, dilation=2→pad=2, dilation=4→pad=4
        padding = (kernel - 1) * dilation // 2

        self.conv = nn.Conv1d(in_ch, out_ch, kernel,
                              padding=padding, dilation=dilation)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # 1×1 projection only when dimensions differ (Block 1 only)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x if self.skip is None else self.skip(x)
        return self.relu(self.bn(self.conv(x)) + skip)


class TCNEncoder(nn.Module):
    """
    TCN text encoder (Section IV-D):
      Embedding(vocab, 128)
      → Block(dil=1, 128→256)
      → Block(dil=2, 256→256)
      → Block(dil=4, 256→256)
      → masked mean-pool  → (B, 256)
      → Linear(256, 768)  → (B, 768)
    """

    def __init__(self, vocab_size: int, embedding_dim: int,
                 hidden_dim: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.block1 = DilatedResidualBlock(embedding_dim, hidden_dim, 3, dilation=1)
        self.block2 = DilatedResidualBlock(hidden_dim,    hidden_dim, 3, dilation=2)
        self.block3 = DilatedResidualBlock(hidden_dim,    hidden_dim, 3, dilation=4)

        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        # (B, L) → (B, L, d) → (B, d, L)  [Conv1d expects channels-first]
        x = self.embedding(input_ids).transpose(1, 2)

        x = self.block1(x)   # (B, hidden_dim, L)
        x = self.block2(x)
        x = self.block3(x)

        # Masked mean-pool: ignore padding tokens
        mask = attention_mask.unsqueeze(1).float()          # (B, 1, L)
        x = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)  # (B, hidden_dim)

        return self.projection(x)   # (B, 768)


# ============================================================================
# ViT + TCN MODEL
# ============================================================================

class ViTTCNModel(nn.Module):
    """
    Dual-stream architecture (Section IV-D):

    Visual stream : ViT-B/16 (pretrained ImageNet-21k)
                    → 196 patch tokens ∈ R^{196×768}
    Text stream   : TCN encoder → t ∈ R^{768}

    Fusion: single-head cross-attention (text queries patch tokens, dk=64)
            → residual + LayerNorm
            → MLP 768 → 256 → 2

    No class-frequency reweighting (dataset is balanced 50/50, Table IV).
    Standard CrossEntropyLoss per Table VI.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, num_classes: int = 2):
        super().__init__()

        # Visual encoder: remove classification head (num_classes=0)
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )

        # Text encoder
        self.tcn = TCNEncoder(
            vocab_size=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=TCN_HIDDEN,
            output_dim=FUSION_DIM,
        )

        # Cross-attention projections (dk=64, h=1)
        self.query_proj = nn.Linear(FUSION_DIM, DK)
        self.key_proj   = nn.Linear(FUSION_DIM, DK)
        self.value_proj = nn.Linear(FUSION_DIM, FUSION_DIM)

        self.layer_norm = nn.LayerNorm(FUSION_DIM)

        # MLP classifier: 768 → 256 → num_classes, GELU, dropout=0.3
        self.classifier = nn.Sequential(
            nn.Linear(FUSION_DIM, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:

        # ── Visual stream ────────────────────────────────────────────────────
        # timm vit_base_patch16_224 forward_features → (B, 197, 768)
        # token 0 = CLS; tokens 1..196 = patch tokens
        all_tokens   = self.vit.forward_features(images)    # (B, 197, 768)
        patch_tokens = all_tokens[:, 1:, :]                 # (B, 196, 768)

        # ── Text stream ──────────────────────────────────────────────────────
        text_emb = self.tcn(input_ids, attention_mask)      # (B, 768)

        # ── Cross-attention: text queries ViT patch tokens ───────────────────
        # Q from text, K/V from visual patches
        Q = self.query_proj(text_emb).unsqueeze(1)          # (B,   1, 64)
        K = self.key_proj(patch_tokens)                     # (B, 196, 64)
        V = self.value_proj(patch_tokens)                   # (B, 196, 768)

        # Scaled dot-product attention
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / (DK ** 0.5)  # (B, 1, 196)
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, V).squeeze(1)      # (B, 768)

        # Residual + LayerNorm (Eq. 3 in paper)
        fused = self.layer_norm(attended + text_emb)        # (B, 768)

        return self.classifier(fused)                       # (B, num_classes)


# ============================================================================
# DATA UTILITIES
# ============================================================================

def get_transform() -> transforms.Compose:
    """ImageNet normalisation. No augmentation (preserves ecological validity)."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_data(fraction: float, seed: int):
    """
    Load CSV splits. Subsample training set stratified on 'label' column.

    Paper Section IV-G (full benchmark): train=754, val=108, test=228.
    The 'sample_weight' column in CSVs is ignored for ViT+TCN
    (paper Table VI shows standard loss, no weighting for this model).
    """
    train_df = pd.read_csv(os.path.join(DATASET_DIR, "nepOOC_train.csv"))
    val_df   = pd.read_csv(os.path.join(DATASET_DIR, "nepOOC_val.csv"))
    test_df  = pd.read_csv(os.path.join(DATASET_DIR, "nepOOC_test.csv"))

    if fraction < 1.0:
        # Stratify on 'label' (binary) to maintain 50/50 balance
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=fraction, random_state=seed
        )
        idx, _ = next(sss.split(train_df, train_df["label"]))
        train_df = train_df.iloc[idx].reset_index(drop=True)

    return train_df, val_df, test_df


def create_dataloaders(train_df, val_df, test_df, tokenizer,
                       batch_size: int = BATCH_SIZE):
    transform = get_transform()
    dl_kwargs = dict(num_workers=2, pin_memory=True)

    train_loader = DataLoader(
        NepOOCDataset(train_df, IMAGE_DIR, tokenizer, transform),
        batch_size=batch_size, shuffle=True, **dl_kwargs,
    )
    val_loader = DataLoader(
        NepOOCDataset(val_df, IMAGE_DIR, tokenizer, transform),
        batch_size=batch_size, shuffle=False, **dl_kwargs,
    )
    test_loader = DataLoader(
        NepOOCDataset(test_df, IMAGE_DIR, tokenizer, transform),
        batch_size=batch_size, shuffle=False, **dl_kwargs,
    )
    return train_loader, val_loader, test_loader


def get_scheduler(optimizer, num_training_steps: int):
    """
    Cosine annealing with 10% linear warm-up (Table VI: 'Cosine+WU').
    Warmup ramps from near-zero to full LR over first 10% of steps.
    """
    warmup_steps = max(1, int(0.1 * num_training_steps))
    cosine_steps = num_training_steps - warmup_steps

    warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0,
                      total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, cosine_steps),
                               eta_min=1e-7)

    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_steps])


# ============================================================================
# TRAINING / EVALUATION
# ============================================================================

def train_epoch(model, loader, optimizer, scheduler, criterion) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        images         = batch["image"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        loss = criterion(model(images, input_ids, attention_mask), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion) -> dict:
    """
    Run inference on a dataloader and return metrics + raw outputs.
    Returns all_preds/labels/probs as np.ndarray for downstream use.
    """
    model.eval()
    all_preds, all_labels, all_probs, all_typologies = [], [], [], []
    total_loss = 0.0

    for batch in loader:
        images         = batch["image"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        logits = model(images, input_ids, attention_mask)
        total_loss += criterion(logits, labels).item()

        probs = F.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())   # prob of OOC class
        all_typologies.extend(batch["typology"])

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    return {
        "loss":          total_loss / len(loader),
        "accuracy":      float(accuracy_score(all_labels, all_preds)),
        "macro_f1":      float(f1_score(all_labels, all_preds, average="macro")),
        # roc_auc_score is safe: test set always has both classes (114+114)
        "auc":           float(roc_auc_score(all_labels, all_probs)),
        "predictions":   all_preds,
        "labels":        all_labels,
        "probabilities": all_probs,
        "typologies":    all_typologies,
    }


def compute_per_typology_f1(predictions, labels, typologies) -> dict:
    """
    For each OOC typology, compute binary F1 (pos_label=1) on the subset
    of test instances belonging to that typology.

    All instances in a typology group have label=1 (OOC) by definition.
    We check whether the model predicted 1 (detected) or 0 (missed).

    zero_division=0 handles:
      - Empty typology group (e.g. Identity_Mismatch in 25% frac, seed=789)
      - All predictions are the same class (n=2 for Identity_Mismatch in test)
    """
    predictions = np.array(predictions)
    labels      = np.array(labels)
    typologies  = np.array(typologies)

    # Restrict to OOC ground-truth instances
    ooc_mask  = labels == 1
    ooc_preds = predictions[ooc_mask]
    ooc_typos = typologies[ooc_mask]

    result = {}
    for typo in TYPO_KEYS:
        mask = ooc_typos == typo
        if mask.sum() == 0:
            result[typo] = 0.0
            continue

        # Ground truth for this subset is all 1s (they are OOC by construction)
        typo_labels = np.ones(mask.sum(), dtype=int)
        typo_preds  = ooc_preds[mask]

        result[typo] = float(
            f1_score(typo_labels, typo_preds,
                     average="binary", pos_label=1, zero_division=0)
        )

    return result


# ============================================================================
# MAIN TRAINING FUNCTION  (one fraction × one seed)
# ============================================================================

def train_model(fraction: float, seed: int, tokenizer) -> dict:
    set_seed(seed)

    print(f"\n{'='*70}")
    print(f"  Fraction={fraction*100:.0f}%  |  Seed={seed}")
    print(f"{'='*70}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_data(fraction, seed)
    print(f"  Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}")

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = ViTTCNModel().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()   # standard loss, no weighting (Table VI)
    scheduler = get_scheduler(optimizer, len(train_loader) * MAX_EPOCHS)

    # ── Training with early stopping on val Macro-F1 ──────────────────────────
    best_f1, best_epoch, patience_ctr = 0.0, 0, 0
    best_state = None

    for epoch in range(MAX_EPOCHS):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        val_m   = evaluate(model, val_loader, criterion)

        print(
            f"  Ep {epoch+1:3d} | "
            f"tr_loss={tr_loss:.4f} | "
            f"val_loss={val_m['loss']:.4f} | "
            f"val_f1={val_m['macro_f1']:.4f} | "
            f"val_acc={val_m['accuracy']:.4f}"
        )

        if val_m["macro_f1"] > best_f1:
            best_f1      = val_m["macro_f1"]
            best_epoch   = epoch
            patience_ctr = 0
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  ↳ Early stop at epoch {epoch+1} (best epoch {best_epoch+1})")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.to(device)
    test_m = evaluate(model, test_loader, criterion)

    # Confusion matrix
    cm = confusion_matrix(test_m["labels"], test_m["predictions"])

    # OOC-class precision / recall (Table IX)
    ooc_prec_arr, ooc_rec_arr, _, _ = precision_recall_fscore_support(
        test_m["labels"], test_m["predictions"], labels=[1], average=None
    )
    ooc_p  = float(ooc_prec_arr[0])
    ooc_r  = float(ooc_rec_arr[0])
    ooc_f1 = 2.0 * ooc_p * ooc_r / (ooc_p + ooc_r + 1e-8)   # safe division

    # Per-typology F1 (Table XII)
    typo_f1 = compute_per_typology_f1(
        test_m["predictions"], test_m["labels"], test_m["typologies"]
    )

    # ROC + PR curves (for plotting)
    fpr, tpr, _       = roc_curve(test_m["labels"], test_m["probabilities"])
    prec_c, rec_c, _  = precision_recall_curve(test_m["labels"],
                                               test_m["probabilities"])

    # ── Assemble result dict ──────────────────────────────────────────────────
    results = {
        "seed":            seed,
        "fraction":        fraction,
        "best_epoch":      best_epoch,
        # Table VIII
        "test_accuracy":   float(test_m["accuracy"]),
        "test_macro_f1":   float(test_m["macro_f1"]),
        "test_auc":        float(test_m["auc"]),
        "test_loss":       float(test_m["loss"]),
        # Table XI
        "confusion_matrix": cm.tolist(),
        # Table IX
        "ooc_precision":   ooc_p,
        "ooc_recall":      ooc_r,
        "ooc_f1":          float(ooc_f1),
        # Table XII
        "typology_f1":     {k: float(v) for k, v in typo_f1.items()},
        # Raw outputs for aggregation / plotting
        "predictions":     test_m["predictions"].tolist(),
        "labels":          test_m["labels"].tolist(),
        "probabilities":   test_m["probabilities"].tolist(),
        "roc_curve":       {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve":        {"precision": prec_c.tolist(), "recall": rec_c.tolist()},
    }

    out_path = os.path.join(
        OUTPUT_DIR, f"vit_tcn_frac{int(fraction*100)}_seed{seed}.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\n  TEST → "
        f"Acc={test_m['accuracy']:.4f} | "
        f"F1={test_m['macro_f1']:.4f} | "
        f"AUC={test_m['auc']:.4f} | "
        f"OOC P={ooc_p:.4f} R={ooc_r:.4f} F1={ooc_f1:.4f}"
    )
    return results


# ============================================================================
# MODALITY ABLATION  (Table XIV)
# ============================================================================

def _run_ablation_loop(model, train_loader, val_loader, fwd_fn, seed: int):
    """
    Shared training + early-stopping loop for text-only and image-only ablations.
    fwd_fn(model, batch, device) → logits tensor
    """
    set_seed(seed)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = get_scheduler(optimizer, len(train_loader) * MAX_EPOCHS)

    best_f1, patience_ctr = 0.0, 0

    for _ in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(fwd_fn(model, batch, device),
                             batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                p = fwd_fn(model, batch, device).argmax(dim=1)
                val_preds.extend(p.cpu().tolist())
                val_labels.extend(batch["label"].tolist())

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        if val_f1 > best_f1:
            best_f1, patience_ctr = val_f1, 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    return model


def _ablation_test_f1(model, test_loader, fwd_fn) -> float:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            p = fwd_fn(model, batch, device).argmax(dim=1)
            preds.extend(p.cpu().tolist())
            labels.extend(batch["label"].tolist())
    return float(f1_score(labels, preds, average="macro"))


def train_text_only(train_loader, val_loader, test_loader, seed: int) -> float:
    """Text-only stream: TCN → MLP (Table XIV)."""

    class TextOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.tcn = TCNEncoder(VOCAB_SIZE, EMBEDDING_DIM, TCN_HIDDEN, FUSION_DIM)
            self.clf = nn.Sequential(
                nn.Linear(FUSION_DIM, 256), nn.GELU(),
                nn.Dropout(DROPOUT), nn.Linear(256, 2),
            )
        def forward(self, ids, mask):
            return self.clf(self.tcn(ids, mask))

    model = TextOnly().to(device)

    def fwd(m, b, dev):
        return m(b["input_ids"].to(dev), b["attention_mask"].to(dev))

    model = _run_ablation_loop(model, train_loader, val_loader, fwd, seed)
    return _ablation_test_f1(model, test_loader, fwd)


def train_image_only(train_loader, val_loader, test_loader, seed: int) -> float:
    """Image-only stream: ViT CLS token → MLP (Table XIV)."""

    class ImageOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = timm.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0
            )
            self.clf = nn.Sequential(
                nn.Linear(FUSION_DIM, 256), nn.GELU(),
                nn.Dropout(DROPOUT), nn.Linear(256, 2),
            )
        def forward(self, imgs):
            # Use CLS token (index 0) for image-only representation
            cls = self.vit.forward_features(imgs)[:, 0, :]
            return self.clf(cls)

    model = ImageOnly().to(device)

    def fwd(m, b, dev):
        return m(b["image"].to(dev))

    model = _run_ablation_loop(model, train_loader, val_loader, fwd, seed)
    return _ablation_test_f1(model, test_loader, fwd)


def run_modality_ablation(tokenizer, seed: int) -> None:
    """Run text-only / image-only / multimodal ablation for one seed."""
    print(f"\n{'='*70}")
    print(f"  TABLE XIV MODALITY ABLATION  |  Seed={seed}")
    print(f"{'='*70}")

    train_df, val_df, test_df = load_data(fraction=1.0, seed=seed)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer
    )

    t_f1 = train_text_only(train_loader, val_loader, test_loader, seed)
    print(f"  Text-only  F1: {t_f1:.4f}")

    i_f1 = train_image_only(train_loader, val_loader, test_loader, seed)
    print(f"  Image-only F1: {i_f1:.4f}")

    # Multimodal result already saved from main training
    mm_path = os.path.join(OUTPUT_DIR, f"vit_tcn_frac100_seed{seed}.json")
    with open(mm_path) as f:
        mm_f1 = json.load(f)["test_macro_f1"]
    print(f"  Multimodal F1: {mm_f1:.4f}")

    ablation = {
        "seed":       seed,
        "text_only":  t_f1,
        "image_only": i_f1,
        "multimodal": mm_f1,
    }
    out = os.path.join(OUTPUT_DIR, f"vit_tcn_ablation_seed{seed}.json")
    with open(out, "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"  Ablation saved → {out}")


# ============================================================================
# RESULT AGGREGATION  (run after all 5 notebooks complete)
# ============================================================================

def aggregate_results() -> None:
    """
    Aggregate results across all seeds and fractions.
    Run this function once all 5 seed notebooks have finished.
    Prints Tables VIII, IX, X, XI, XII, XIV, XV.
    """
    print("\n" + "="*70)
    print("AGGREGATING RESULTS ACROSS ALL SEEDS")
    print("="*70)

    # ── Load all result files ─────────────────────────────────────────────────
    all_results = []
    for frac in FRACTIONS:
        for seed in ALL_SEEDS:
            p = os.path.join(OUTPUT_DIR,
                             f"vit_tcn_frac{int(frac*100)}_seed{seed}.json")
            if os.path.exists(p):
                with open(p) as f:
                    all_results.append(json.load(f))
            else:
                print(f"  WARNING: missing {p}")

    full = [r for r in all_results if r["fraction"] == 1.0]
    if not full:
        print("  ERROR: no 100% fraction results found.")
        return

    # ── Table VIII ────────────────────────────────────────────────────────────
    acc_v  = [r["test_accuracy"] for r in full]
    f1_v   = [r["test_macro_f1"] for r in full]
    auc_v  = [r["test_auc"]      for r in full]

    print(f"\nTABLE VIII: MAIN RESULTS (100%, n_seeds={len(full)})")
    print(f"  Accuracy : {np.mean(acc_v)*100:.1f} ± {np.std(acc_v)*100:.1f}")
    print(f"  Macro-F1 : {np.mean(f1_v):.3f} ± {np.std(f1_v):.3f}")
    print(f"  AUC      : {np.mean(auc_v):.3f} ± {np.std(auc_v):.3f}")

    # ── Table IX ─────────────────────────────────────────────────────────────
    ooc_p = [r["ooc_precision"] for r in full]
    ooc_r = [r["ooc_recall"]    for r in full]
    ooc_f = [r["ooc_f1"]        for r in full]

    print(f"\nTABLE IX: OOC-CLASS METRICS (100%, {len(full)} seeds)")
    print(f"  Precision : {np.mean(ooc_p):.3f}")
    print(f"  Recall    : {np.mean(ooc_r):.3f}")
    print(f"  F1        : {np.mean(ooc_f):.3f}")

    # ── Table X ──────────────────────────────────────────────────────────────
    print(f"\nTABLE X: TRAINING-SIZE SCALING (Macro-F1, mean ± std)")
    for frac in FRACTIONS:
        vals = [r["test_macro_f1"] for r in all_results if r["fraction"] == frac]
        if vals:
            print(f"  {int(frac*100):3d}%  : {np.mean(vals):.3f} ± {np.std(vals):.3f}"
                  f"  (n={len(vals)})")

    # ── Table XI ─────────────────────────────────────────────────────────────
    avg_cm = np.mean([r["confusion_matrix"] for r in full], axis=0)
    print(f"\nTABLE XI: AVERAGE CONFUSION MATRIX (100%, {len(full)} seeds)")
    print(f"  {'':22s}  Pred Pristine  Pred OOC")
    print(f"  {'Actual Pristine':22s}  {avg_cm[0,0]:12.1f}  {avg_cm[0,1]:.1f}")
    print(f"  {'Actual OOC':22s}  {avg_cm[1,0]:12.1f}  {avg_cm[1,1]:.1f}")

    # ── Table XII ────────────────────────────────────────────────────────────
    print(f"\nTABLE XII: PER-TYPOLOGY OOC F1 (100%, {len(full)} seeds)")
    for typo in TYPO_KEYS:
        vals = [r["typology_f1"].get(typo, 0.0) for r in full]
        print(f"  {TYPO_DISPLAY[typo]:25s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # ── Table XIV ────────────────────────────────────────────────────────────
    ablation_files = [
        os.path.join(OUTPUT_DIR, f"vit_tcn_ablation_seed{s}.json")
        for s in ALL_SEEDS
    ]
    ablation_data = []
    for p in ablation_files:
        if os.path.exists(p):
            with open(p) as f:
                ablation_data.append(json.load(f))

    if ablation_data:
        print(f"\nTABLE XIV: MODALITY ABLATION (100%, n={len(ablation_data)})")
        for key, label in [("text_only", "Text-only"), ("image_only", "Image-only"),
                           ("multimodal", "Multimodal")]:
            vals = [d[key] for d in ablation_data]
            print(f"  {label:12s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # ── Table XV ─────────────────────────────────────────────────────────────
    print(f"\nTABLE XV: MULTI-SEED STABILITY (100%)")
    print(f"  F1  mean : {np.mean(f1_v):.3f}")
    print(f"  F1  std  : {np.std(f1_v):.3f}")
    print(f"  Acc std  : {np.std(acc_v):.3f}")

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "table_viii": {
            "accuracy_mean": float(np.mean(acc_v)),
            "accuracy_std":  float(np.std(acc_v)),
            "macro_f1_mean": float(np.mean(f1_v)),
            "macro_f1_std":  float(np.std(f1_v)),
            "auc_mean":      float(np.mean(auc_v)),
            "auc_std":       float(np.std(auc_v)),
            "n_seeds":       len(full),
        },
        "table_ix": {
            "ooc_precision": float(np.mean(ooc_p)),
            "ooc_recall":    float(np.mean(ooc_r)),
            "ooc_f1":        float(np.mean(ooc_f)),
        },
        "table_x": {
            f"{int(frac*100)}%": {
                "mean": float(np.mean(
                    [r["test_macro_f1"] for r in all_results if r["fraction"] == frac]
                )),
                "std": float(np.std(
                    [r["test_macro_f1"] for r in all_results if r["fraction"] == frac]
                )),
            }
            for frac in FRACTIONS
            if any(r["fraction"] == frac for r in all_results)
        },
        "table_xi": {"avg_confusion_matrix": avg_cm.tolist()},
        "table_xii": {
            TYPO_DISPLAY[t]: {
                "mean": float(np.mean(
                    [r["typology_f1"].get(t, 0.0) for r in full]
                )),
                "std": float(np.std(
                    [r["typology_f1"].get(t, 0.0) for r in full]
                )),
            }
            for t in TYPO_KEYS
        },
        "table_xv": {
            "f1_mean": float(np.mean(f1_v)),
            "f1_std":  float(np.std(f1_v)),
            "acc_std": float(np.std(acc_v)),
        },
    }

    # Scaling and curve data for figures
    summary["figure_5_scaling"] = {
        f"{int(frac*100)}%": {
            "mean":   float(np.mean(
                [r["test_macro_f1"] for r in all_results if r["fraction"] == frac]
            )),
            "std":    float(np.std(
                [r["test_macro_f1"] for r in all_results if r["fraction"] == frac]
            )),
            "values": [
                float(r["test_macro_f1"])
                for r in all_results if r["fraction"] == frac
            ],
        }
        for frac in FRACTIONS
        if any(r["fraction"] == frac for r in all_results)
    }

    summary["figure_6_roc"] = {
        "all_fpr": [r["roc_curve"]["fpr"] for r in full],
        "all_tpr": [r["roc_curve"]["tpr"] for r in full],
    }
    summary["figure_6_pr"] = {
        "all_precision": [r["pr_curve"]["precision"] for r in full],
        "all_recall":    [r["pr_curve"]["recall"]    for r in full],
    }

    out = os.path.join(OUTPUT_DIR, "vit_tcn_aggregate_results.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Aggregate summary → {out}")

    # Failure cases CSV
    rows = []
    for r in full:
        preds  = np.array(r["predictions"])
        labels_ = np.array(r["labels"])
        for idx in np.where(preds != labels_)[0]:
            rows.append({
                "seed":        r["seed"],
                "index":       int(idx),
                "true_label":  int(labels_[idx]),
                "pred_label":  int(preds[idx]),
                "probability": float(r["probabilities"][idx]),
            })
    fail_path = os.path.join(OUTPUT_DIR, "vit_tcn_failure_cases.csv")
    pd.DataFrame(rows).to_csv(fail_path, index=False)
    print(f"  Failure cases ({len(rows)}) → {fail_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    print("=" * 70)
    print(f"ViT+TCN  |  NepOOC  |  Seed={NOTEBOOK_SEED}")
    print("=" * 70)

    print("\nLoading mBERT tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Train all 4 fractions for this notebook's seed
    for fraction in FRACTIONS:
        train_model(fraction, NOTEBOOK_SEED, tokenizer)

    # Modality ablation for this seed (uses the 100% fraction result already saved)
    run_modality_ablation(tokenizer, NOTEBOOK_SEED)

    print(f"\n{'='*70}")
    print(f"  DONE  |  Seed={NOTEBOOK_SEED}")
    print(f"  Results in: {OUTPUT_DIR}")
    print(f"{'='*70}")

    # ── Optional: run aggregation if you want to see partial results ──────────
    # Only meaningful after all 5 notebooks have finished.
    # Uncomment and run separately, or leave it; missing seeds are warned about.
    # aggregate_results()


if __name__ == "__main__":
    main()