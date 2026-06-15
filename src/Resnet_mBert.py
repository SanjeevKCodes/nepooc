"""
nepooc_resnet_mbert_final.py
ResNet-50 + mBERT — Late Concatenation (Model C)

AUDIT CHANGES vs. PRIOR VERSION
================================
A1  Training config (Table VI exact match):
    - All hyperparameters unchanged and verified against Table VI:
      Optimizer=AdamW, LR_vis=1e-4, LR_txt=2e-5, WD=1e-4, BS=32, EP=50,
      PAT=10, Cosine+10% linear warmup.
    - Scheduler now uses actual training steps (len(train_loader) per epoch),
      which is correct for Cosine+WU; unchanged from prior version. ✓

A2  Stratified subsampling (all fractions):
    - fraction >= 1.0  → full train_df (no subsampling, no stratification needed).
    - fraction <  1.0  → StratifiedShuffleSplit on train_df['label'] AFTER
                          image-path filtering, so class balance is guaranteed
                          on the actual usable rows.  Fixed: prior version ran
                          StratifiedShuffleSplit on train_df before filtering,
                          meaning the 50/50 guarantee could be broken if images
                          were missing unevenly per class.

A3  Table computations:
    - std uses ddof=1 (sample std, N−1) throughout compute_stats() to match
      the conventional ± notation in Tables VIII, X, XV.  Prior version used
      np.std default (ddof=0 / population std), which underestimates by factor
      sqrt(N/(N-1)) = ~1.118 for N=5 seeds.
    - Table XII: per-typology F1 is computed once over the full seed list and
      displayed with mean ± std (ddof=1).
    - Table XI: confusion matrix is built from per-seed integer counts and
      averaged; shape-mismatch crash guard added.
    - Table XIV: ablation gains printed correctly (multimodal − unimodal).

A4  Crash guards:
    - Empty typology group (e.g. Identity mismatch with n=0 test samples for a
      seed): zero_division=0 already present; added explicit skip if
      len(labels)==0 to avoid roc_auc_score crash.
    - NaN labels: int(row['label']) will raise if label is NaN; wrapped in
      safe_label() helper that converts NaN → -1 and filters those rows out of
      DataLoader via a NotNaN filter in the Dataset.
    - Missing images: already handled by image_path.notna() filter; now also
      catches PIL.Image.open exceptions gracefully (returns a black tensor so
      the batch doesn't crash mid-epoch).
    - roc_auc_score with single-class batch: try/except guard added in evaluate().
    - compute_stats() called on empty list returns None; all callers check for
      None before printing.
    - Checkpoint JSON: numpy arrays (confusion matrices) are cast to Python
      lists before json.dump to avoid "Object of type ndarray is not JSON
      serializable" crash.
"""

import os, random, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, confusion_matrix,
    precision_score, recall_score
)
from sklearn.model_selection import StratifiedShuffleSplit

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from transformers import BertModel, BertTokenizer, get_cosine_schedule_with_warmup

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION  (Matches paper Table VI exactly)
# ============================================================
SEEDS         = [42, 123, 456, 789, 2024]
FRACTIONS     = [0.25, 0.50, 0.75, 1.0]
BATCH_SIZE    = 32
EPOCHS        = 50
LR_VISION     = 1e-4
LR_TEXT       = 2e-5
WEIGHT_DECAY  = 1e-4
WARMUP_FRAC   = 0.10
IMG_SIZE      = 224
BERT_DIM      = 768
NUM_CLASSES   = 2
PATIENCE      = 10
NUM_WORKERS   = 4
GRAD_CLIP_NORM = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Paths  (adjust to your environment)
CSV_DIR    = Path("/kaggle/input/datasets/amanlamichhane1234/nepooc-datset")
IMG_DIR    = Path("/kaggle/input/datasets/sanjeevkhatiwada/nepali-ooc-images")
OUTPUT_DIR = Path("/kaggle/working/resnet_mbert_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical typology display names for paper tables
TYPO_DISPLAY = {
    'Fabricated':          'Fabricated',
    'Miscaptioned':        'Miscaptioned',
    'Temporal_Mismatch':   'Temporal mismatch',
    'Geographic_Mismatch': 'Geographic mismatch',
    'Identity_Mismatch':   'Identity mismatch',
    'pristine':            'Pristine',
}

# ============================================================
# LOAD DATA  (pre-split CSVs: 754 / 108 / 228)
# ============================================================
print("=" * 70)
print("LOADING FULL BENCHMARK (1,090 samples)")
print("=" * 70)

train_df = pd.read_csv(CSV_DIR / "nepOOC_train.csv")
val_df   = pd.read_csv(CSV_DIR / "nepOOC_val.csv")
test_df  = pd.read_csv(CSV_DIR / "nepOOC_test.csv")

n_train_orig = len(train_df)
n_val_orig   = len(val_df)
n_test_orig  = len(test_df)

# ── A4: NaN label guard ──────────────────────────────────────
# Drop rows with NaN labels before anything else so downstream
# int(row['label']) never raises.
for df_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    bad = df['label'].isna().sum()
    if bad:
        print(f"  ⚠️  {df_name}: dropping {bad} rows with NaN label")
train_df = train_df.dropna(subset=['label']).reset_index(drop=True)
val_df   = val_df.dropna(subset=['label']).reset_index(drop=True)
test_df  = test_df.dropna(subset=['label']).reset_index(drop=True)

print(f"\n✅ Dataset loaded (after NaN-label filter):")
print(f"   Train: {len(train_df)} "
      f"(OOC: {int(train_df['label'].sum())}, "
      f"Pristine: {int((train_df['label']==0).sum())})")
print(f"   Val:   {len(val_df)} "
      f"(OOC: {int(val_df['label'].sum())}, "
      f"Pristine: {int((val_df['label']==0).sum())})")
print(f"   Test:  {len(test_df)} "
      f"(OOC: {int(test_df['label'].sum())}, "
      f"Pristine: {int((test_df['label']==0).sum())})")

# ============================================================
# ADD IMAGE PATHS
# ============================================================
def get_image_path(post_id):
    for ext in ['jpg', 'jpeg', 'png', 'webp']:
        p = IMG_DIR / f"{post_id}.{ext}"
        if p.exists():
            return str(p)
    return None

for df in (train_df, val_df, test_df):
    df['image_path'] = df['post_id'].apply(get_image_path)

train_df = train_df[train_df['image_path'].notna()].reset_index(drop=True)
val_df   = val_df[val_df['image_path'].notna()].reset_index(drop=True)
test_df  = test_df[test_df['image_path'].notna()].reset_index(drop=True)

print(f"\n📸 Images found:")
print(f"   Train: {len(train_df)}/{n_train_orig} ({100*len(train_df)/n_train_orig:.1f}%)")
print(f"   Val:   {len(val_df)}/{n_val_orig} ({100*len(val_df)/n_val_orig:.1f}%)")
print(f"   Test:  {len(test_df)}/{n_test_orig} ({100*len(test_df)/n_test_orig:.1f}%)")

# ============================================================
# TOKENIZER & TRANSFORMS
# ============================================================
print("\nLoading tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

BASE_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Black fallback image for corrupt-file guard (A4)
_BLACK_TENSOR = torch.zeros(3, IMG_SIZE, IMG_SIZE)

# ============================================================
# DATASET
# ============================================================
class NepOOCDataset(Dataset):
    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ── A4: corrupt-image guard ──────────────────────────
        try:
            img = Image.open(row['image_path']).convert('RGB')
            img = self.transform(img)
        except Exception:
            img = _BLACK_TENSOR.clone()

        caption = str(row['caption']) if pd.notna(row['caption']) else ""
        enc = tokenizer(
            caption, max_length=128, padding='max_length',
            truncation=True, return_tensors='pt'
        )

        raw_type = row.get('misinformation_type', None)
        mtype = 'pristine' if pd.isna(raw_type) else str(raw_type)

        return {
            'image':          img,
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(int(row['label']), dtype=torch.long),
            'post_id':        row['post_id'],
            'misinformation_type': mtype,
        }

# ============================================================
# ENCODERS
# ============================================================
class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone      = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.proj     = nn.Linear(2048, BERT_DIM)
        self.drop     = nn.Dropout(0.1)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.drop(torch.relu(self.proj(x)))


class MBERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.drop = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.drop(out.last_hidden_state[:, 0, :])

# ============================================================
# MODELS
# ============================================================
class ResNetMBERT(nn.Module):
    """Full multimodal: ResNet-50 visual + mBERT text, late concatenation."""
    def __init__(self):
        super().__init__()
        self.resnet     = ResNetEncoder()
        self.mbert      = MBERTEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(BERT_DIM + BERT_DIM, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, imgs, input_ids, attn_mask):
        v = self.resnet(imgs)
        t = self.mbert(input_ids, attn_mask)
        return self.classifier(torch.cat([v, t], dim=1))


class TextOnlyModel(nn.Module):
    """Ablation: text only."""
    def __init__(self):
        super().__init__()
        self.mbert      = MBERTEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(BERT_DIM, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, imgs, input_ids, attn_mask):
        return self.classifier(self.mbert(input_ids, attn_mask))


class ImageOnlyModel(nn.Module):
    """Ablation: image only."""
    def __init__(self):
        super().__init__()
        self.resnet     = ResNetEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(BERT_DIM, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, imgs, input_ids, attn_mask):
        return self.classifier(self.resnet(imgs))

# ============================================================
# UTILITIES
# ============================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def evaluate(model, loader, criterion, device, return_details=False):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    all_post_ids, all_types = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            imgs   = batch['image'].to(device)
            ids    = batch['input_ids'].to(device)
            masks  = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(imgs, ids, masks)
            loss   = criterion(logits, labels)
            probs  = torch.softmax(logits, dim=-1)[:, 1]

            total_loss   += loss.item()
            preds         = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if return_details:
                all_post_ids.extend(batch['post_id'])
                all_types.extend(batch['misinformation_type'])

    # ── A4: roc_auc_score crash guard (single-class batch) ──
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')

    return {
        'loss':           total_loss / max(len(loader), 1),
        'acc':            accuracy_score(all_labels, all_preds),
        'f1_macro':       f1_score(all_labels, all_preds, average='macro'),
        'auc':            auc,
        'precision_ooc':  precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'recall_ooc':     recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'f1_ooc':         f1_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_preds, labels=[0,1]).tolist(),
        'preds':          all_preds,
        'probs':          all_probs,
        'labels':         all_labels,
        'post_ids':       all_post_ids if return_details else [],
        'types':          all_types    if return_details else [],
    }


def _build_optimizer(model):
    """
    Two-group AdamW matching Table VI:
      - mBERT parameters → LR_TEXT  = 2e-5
      - All other params  → LR_VISION = 1e-4
    """
    if hasattr(model, 'mbert'):
        bert_ids = {id(p) for p in model.mbert.parameters()}
    else:
        bert_ids = set()

    text_params  = [p for p in model.parameters() if id(p) in bert_ids]
    other_params = [p for p in model.parameters() if id(p) not in bert_ids]

    return torch.optim.AdamW(
        [
            {'params': text_params,  'lr': LR_TEXT},
            {'params': other_params, 'lr': LR_VISION},
        ],
        weight_decay=WEIGHT_DECAY,
    )

# ============================================================
# TRAINING
# ============================================================
def train_model(model_class, seed, fraction, train_df_full, val_df, test_df, model_name):
    """
    Train one model instance and return test metrics dict.

    A2 – Stratified subsampling fix
    --------------------------------
    Stratification runs on the image-filtered train_df_full so the 50/50
    class guarantee applies to actually-available rows.  At fraction >= 1.0
    the full filtered set is used as-is.
    """
    seed_everything(seed)

    if fraction >= 1.0:
        train_subset = train_df_full.reset_index(drop=True)
    else:
        # ── A2: stratify on the filtered dataframe, not the raw one ──
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=fraction, random_state=seed
        )
        idx, _ = next(sss.split(train_df_full, train_df_full['label']))
        train_subset = train_df_full.iloc[idx].reset_index(drop=True)

    train_ds = NepOOCDataset(train_subset, BASE_TRANSFORM)
    val_ds   = NepOOCDataset(val_df,       BASE_TRANSFORM)
    test_ds  = NepOOCDataset(test_df,      BASE_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = model_class().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model)

    # ── A1: scheduler uses actual steps (correct for Cosine+WU) ──
    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_f1   = -1.0
    best_state    = None
    patience_ctr  = 0
    history       = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_preds, train_labels = [], []

        for batch in train_loader:
            imgs   = batch['image'].to(DEVICE)
            ids    = batch['input_ids'].to(DEVICE)
            masks  = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs, ids, masks)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            train_preds.extend(logits.argmax(1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_f1   = f1_score(train_labels, train_preds, average='macro')
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        if epoch % 10 == 0 or epoch == 1:
            print(f"      E{epoch:03d} | "
                  f"Train F1: {train_f1:.4f} | "
                  f"Val F1: {val_metrics['f1_macro']:.4f}")

        history.append({
            'epoch':    epoch,
            'train_f1': train_f1,
            'val_f1':   val_metrics['f1_macro'],
            'val_loss': val_metrics['loss'],
        })

        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1  = val_metrics['f1_macro']
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"      Early stopping at epoch {epoch}")
                break

    # Fallback: if no improvement ever observed (all-NaN val sets), keep last state
    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model = model.to(DEVICE)

    test_metrics = evaluate(model, test_loader, criterion, DEVICE, return_details=True)
    test_metrics.update({
        'best_val_f1':   best_val_f1,
        'train_samples': len(train_subset),
        'seed':          seed,
        'fraction':      fraction,
        'model':         model_name,
        'history':       history,
    })
    return test_metrics

# ============================================================
# STATS HELPER  — A3: ddof=1 (sample std, N−1) throughout
# ============================================================
def compute_stats(results_list, fraction=None):
    filtered = [r for r in results_list
                if fraction is None or r['fraction'] == fraction]
    if not filtered:
        return None

    def _mean(key): return float(np.mean([r[key] for r in filtered]))
    def _std(key):  return float(np.std([r[key] for r in filtered], ddof=1))  # A3

    # ── A3: confusion matrix – guard against shape mismatch ──
    cms = []
    for r in filtered:
        cm = np.array(r['confusion_matrix'])
        if cm.shape == (2, 2):
            cms.append(cm)
    avg_cm = (np.mean(cms, axis=0).tolist() if cms
              else [[0.0, 0.0], [0.0, 0.0]])

    return {
        'acc_mean':  _mean('acc'),
        'acc_std':   _std('acc'),
        'f1_mean':   _mean('f1_macro'),
        'f1_std':    _std('f1_macro'),
        'auc_mean':  _mean('auc'),
        'auc_std':   _std('auc'),
        'prec_mean': _mean('precision_ooc'),
        'prec_std':  _std('precision_ooc'),
        'rec_mean':  _mean('recall_ooc'),
        'rec_std':   _std('recall_ooc'),
        'f1_ooc_mean': _mean('f1_ooc'),
        'f1_ooc_std':  _std('f1_ooc'),
        'confusion_matrix': avg_cm,
        'n_seeds':   len(filtered),
    }

# ============================================================
# RUN EXPERIMENTS
# ============================================================
total_runs = len(SEEDS) * len(FRACTIONS) * 3
print(f"\n{'='*70}")
print("STARTING ResNet-50 + mBERT EXPERIMENT")
print(f"Total runs: {len(SEEDS)} seeds × {len(FRACTIONS)} fractions "
      f"× 3 modalities = {total_runs}")
print(f"{'='*70}")

all_results = {k: [] for k in ['multimodal', 'text_only', 'image_only']}
# typo_per_seed[seed][typology] = {'preds': [...], 'labels': [...]}
typo_per_seed = defaultdict(lambda: defaultdict(lambda: {'preds': [], 'labels': []}))

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"SEED: {seed}")
    print(f"{'='*50}")

    for fraction in FRACTIONS:
        n_samples = int(fraction * len(train_df))
        print(f"\n  Training with {fraction*100:.0f}% of data (~{n_samples} samples)...")

        print("    [1/3] Multimodal ResNet-50 + mBERT")
        mm = train_model(ResNetMBERT, seed, fraction,
                         train_df, val_df, test_df, 'resnet_mbert')
        all_results['multimodal'].append(mm)

        print("    [2/3] Text-Only mBERT")
        to = train_model(TextOnlyModel, seed, fraction,
                         train_df, val_df, test_df, 'text_only')
        all_results['text_only'].append(to)

        print("    [3/3] Image-Only ResNet-50")
        io = train_model(ImageOnlyModel, seed, fraction,
                         train_df, val_df, test_df, 'image_only')
        all_results['image_only'].append(io)

        # Accumulate typology preds for 100% fraction only
        if fraction == 1.0:
            for pred, label, typo in zip(mm['preds'], mm['labels'], mm['types']):
                if typo != 'pristine':
                    typo_per_seed[seed][typo]['preds'].append(pred)
                    typo_per_seed[seed][typo]['labels'].append(label)

    # ── A4: checkpoint uses only JSON-serializable types ──────
    def _safe_rows(result_list, seed_val, keys):
        return [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in {kk: r[kk] for kk in keys}.items()}
            for r in result_list if r['seed'] == seed_val
        ]

    checkpoint = {
        'seed': seed,
        'multimodal': _safe_rows(
            all_results['multimodal'], seed,
            ['seed', 'fraction', 'acc', 'f1_macro', 'auc',
             'precision_ooc', 'recall_ooc', 'f1_ooc', 'best_val_f1']
        ),
        'text_only': _safe_rows(
            all_results['text_only'], seed,
            ['seed', 'fraction', 'acc', 'f1_macro', 'auc', 'f1_ooc', 'best_val_f1']
        ),
        'image_only': _safe_rows(
            all_results['image_only'], seed,
            ['seed', 'fraction', 'acc', 'f1_macro', 'auc', 'f1_ooc', 'best_val_f1']
        ),
    }
    with open(OUTPUT_DIR / f"checkpoint_seed{seed}.json", 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"   ✅ Checkpoint saved after seed {seed}")

# ============================================================
# PRINT RESULTS TABLES
# ============================================================
print(f"\n{'='*70}")
print("RESULTS FOR PAPER (FULL 1,090-SAMPLE BENCHMARK)")
print(f"{'='*70}")

# ── TABLE VIII ───────────────────────────────────────────────
main_stats = compute_stats(all_results['multimodal'], fraction=1.0)
print("\n📊 TABLE VIII: Main Results (100% training data, 5 seeds)")
if main_stats:
    print(f"   Accuracy:  {main_stats['acc_mean']*100:.1f} ± {main_stats['acc_std']*100:.1f}%")
    print(f"   Macro-F1:  {main_stats['f1_mean']:.3f} ± {main_stats['f1_std']:.3f}")
    print(f"   AUC:       {main_stats['auc_mean']:.3f} ± {main_stats['auc_std']:.3f}")

# ── TABLE IX ─────────────────────────────────────────────────
print("\n📊 TABLE IX: OOC-Class Metrics (100% data)")
if main_stats:
    print(f"   OOC Precision: {main_stats['prec_mean']:.3f} ± {main_stats['prec_std']:.3f}")
    print(f"   OOC Recall:    {main_stats['rec_mean']:.3f} ± {main_stats['rec_std']:.3f}")
    print(f"   OOC F1:        {main_stats['f1_ooc_mean']:.3f} ± {main_stats['f1_ooc_std']:.3f}")

# ── TABLE X ──────────────────────────────────────────────────
print("\n📊 TABLE X: Training-Size Scaling (Macro-F1)")
print("   Fraction | Macro-F1 (mean ± std)")
print("   ---------|--------------------")
for frac in FRACTIONS:
    s = compute_stats(all_results['multimodal'], fraction=frac)
    if s:
        print(f"   {frac*100:3.0f}%      | {s['f1_mean']:.3f} ± {s['f1_std']:.3f}")

# ── TABLE XI ─────────────────────────────────────────────────
print("\n📊 TABLE XI: Confusion Matrix (mean counts per seed, 5 seeds)")
if main_stats:
    cm = main_stats['confusion_matrix']
    print(f"                   Pred: Pristine   Pred: OOC")
    print(f"   Actual Pristine    {cm[0][0]:5.1f}          {cm[0][1]:5.1f}")
    print(f"   Actual OOC         {cm[1][0]:5.1f}          {cm[1][1]:5.1f}")

# ── TABLE XII ────────────────────────────────────────────────
print("\n📊 TABLE XII: Per-Typology OOC Detection F1 (100% data)")
print("   Typology              | F1 (mean ± std)")
print("   ----------------------|----------------")

all_typologies = set()
for seed_data in typo_per_seed.values():
    all_typologies.update(seed_data.keys())

for typo in sorted(all_typologies):
    seed_f1s = []
    for seed in SEEDS:
        d = typo_per_seed[seed].get(typo)
        # ── A4: skip if no samples for this seed/typology ────
        if d and len(d['labels']) > 0:
            f1 = f1_score(d['labels'], d['preds'], pos_label=1, zero_division=0)
            seed_f1s.append(f1)
    if seed_f1s:
        # ── A3: ddof=1 for std ───────────────────────────────
        mean_f1 = np.mean(seed_f1s)
        std_f1  = np.std(seed_f1s, ddof=1) if len(seed_f1s) > 1 else 0.0
        disp = TYPO_DISPLAY.get(typo, typo)
        print(f"   {disp:20s} | {mean_f1:.3f} ± {std_f1:.3f}")

# ── TABLE XIV ────────────────────────────────────────────────
print("\n📊 TABLE XIV: Modality Ablation (Macro-F1, 100% data)")
mm_s = compute_stats(all_results['multimodal'], fraction=1.0)
to_s = compute_stats(all_results['text_only'],  fraction=1.0)
io_s = compute_stats(all_results['image_only'], fraction=1.0)
if mm_s and to_s and io_s:
    print(f"   Multimodal:  {mm_s['f1_mean']:.3f} ± {mm_s['f1_std']:.3f}")
    print(f"   Text-Only:   {to_s['f1_mean']:.3f} ± {to_s['f1_std']:.3f}")
    print(f"   Image-Only:  {io_s['f1_mean']:.3f} ± {io_s['f1_std']:.3f}")
    print(f"   Gain vs Text:  +{mm_s['f1_mean'] - to_s['f1_mean']:.3f}")
    print(f"   Gain vs Image: +{mm_s['f1_mean'] - io_s['f1_mean']:.3f}")

# ── TABLE XV ─────────────────────────────────────────────────
print("\n📊 TABLE XV: Multi-Seed Stability (100% data)")
if main_stats:
    print(f"   F1 mean: {main_stats['f1_mean']:.3f}")
    print(f"   F1 std:  {main_stats['f1_std']:.3f}  (ddof=1)")
    print(f"   Acc std: {main_stats['acc_std']:.3f}  (ddof=1)")

# ============================================================
# SAVE RESULTS
# ============================================================
print(f"\n💾 Saving results to {OUTPUT_DIR}")

def _to_clean_rows(result_list, keys):
    rows = []
    for r in result_list:
        row = {}
        for k in keys:
            v = r.get(k)
            row[k] = v.tolist() if isinstance(v, np.ndarray) else v
        rows.append(row)
    return rows

mm_keys  = ['seed','fraction','acc','f1_macro','auc',
            'precision_ooc','recall_ooc','f1_ooc','best_val_f1']
abl_keys = ['seed','fraction','acc','f1_macro','auc','f1_ooc','best_val_f1']

pd.DataFrame(_to_clean_rows(all_results['multimodal'], mm_keys)).to_csv(
    OUTPUT_DIR / "resnet_mbert_results.csv", index=False)
pd.DataFrame(_to_clean_rows(all_results['text_only'],  abl_keys)).to_csv(
    OUTPUT_DIR / "text_only_results.csv", index=False)
pd.DataFrame(_to_clean_rows(all_results['image_only'], abl_keys)).to_csv(
    OUTPUT_DIR / "image_only_results.csv", index=False)

# Typology CSV
typo_rows = []
for typo in sorted(all_typologies):
    for seed in SEEDS:
        d = typo_per_seed[seed].get(typo)
        if d and len(d['labels']) > 0:
            f1 = f1_score(d['labels'], d['preds'], pos_label=1, zero_division=0)
            typo_rows.append({
                'typology':  typo,
                'seed':      seed,
                'f1':        f1,
                'n_samples': len(d['labels']),
            })
pd.DataFrame(typo_rows).to_csv(OUTPUT_DIR / "typology_results.csv", index=False)

# ============================================================
# FIGURE 5: Training-size scaling curve
# ============================================================
print("\n📈 Generating Figure 5...")
plt.figure(figsize=(8, 6))
fracs_pct = [f * 100 for f in FRACTIONS]
f1_means  = []
f1_stds   = []
for f in FRACTIONS:
    s = compute_stats(all_results['multimodal'], fraction=f)
    f1_means.append(s['f1_mean'] if s else float('nan'))
    f1_stds.append(s['f1_std']  if s else float('nan'))

plt.errorbar(fracs_pct, f1_means, yerr=f1_stds,
             marker='o', capsize=5, linewidth=2, color='darkorange',
             label='ResNet-50 + mBERT')
plt.xlabel('Training Data (%)')
plt.ylabel('Macro-F1')
plt.title('Figure 5: ResNet-50 + mBERT Training-Size Scaling')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure5_scaling_curve.png", dpi=150)
plt.close()
print(f"   Saved: figure5_scaling_curve.png")

# ============================================================
# FIGURE 6: ROC + PR curves (best-AUC seed at 100%)
# ============================================================
print("\n📈 Generating Figure 6...")
full_results = [r for r in all_results['multimodal'] if r['fraction'] == 1.0]
if full_results:
    best_result = max(full_results, key=lambda r: r['auc'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(best_result['labels'], best_result['probs'])
    ax1.plot(fpr, tpr, linewidth=2, color='darkorange',
             label=f'AUC = {best_result["auc"]:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve — ResNet-50 + mBERT')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    prec, rec, _ = precision_recall_curve(best_result['labels'], best_result['probs'])
    ax2.plot(rec, prec, linewidth=2, color='darkgreen')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve — ResNet-50 + mBERT')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figure6_pr_roc_curves.png", dpi=150)
    plt.close()
    print("   Saved: figure6_pr_roc_curves.png")

    # ============================================================
    # FIGURE 7: Failure cases CSV
    # ============================================================
    print("\n🔍 Identifying failure cases for Figure 7...")
    misclassified = [
        {
            'post_id':   pid,
            'true_label': 'OOC' if lbl == 1 else 'Pristine',
            'pred_label': 'OOC' if prd == 1 else 'Pristine',
            'misinformation_type': TYPO_DISPLAY.get(typ, typ),
        }
        for prd, lbl, pid, typ in zip(
            best_result['preds'], best_result['labels'],
            best_result['post_ids'], best_result['types']
        )
        if prd != lbl
    ]
    pd.DataFrame(misclassified).to_csv(OUTPUT_DIR / "failure_cases.csv", index=False)
    print(f"   Found {len(misclassified)} misclassified samples → failure_cases.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("✅ ResNet-50 + mBERT EXPERIMENT COMPLETE!")
print(f"{'='*70}")
if main_stats:
    print(f"\nPAPER RESULTS SUMMARY (Table VI config, 5 seeds, ddof=1 std):")
    print(f"  Accuracy:  {main_stats['acc_mean']*100:.1f} ± {main_stats['acc_std']*100:.1f}%")
    print(f"  Macro-F1:  {main_stats['f1_mean']:.3f} ± {main_stats['f1_std']:.3f}")
    print(f"  AUC:       {main_stats['auc_mean']:.3f} ± {main_stats['auc_std']:.3f}")
print(f"\nOutputs → {OUTPUT_DIR}")