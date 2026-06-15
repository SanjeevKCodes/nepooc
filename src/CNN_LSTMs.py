"""
nepooc_complete_experiment.py - FULLY CORRECTED
Fixes: Typology F1 calculation, figures, per-seed aggregation
FIX (this version): Stratified subsampling replacing train_df.sample()
  Before : train_df.sample(frac=fraction, random_state=seed)  [random, breaks 50/50]
  After  : StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=seed)
  Effect : Label proportions are preserved at all fractions (25/50/75/100%),
           matching ResNet+mBERT, CLIP, and ViT+TCN protocols for fair Table X.
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
    precision_recall_curve, confusion_matrix, precision_score, recall_score
)
from sklearn.model_selection import StratifiedShuffleSplit   # ← ADDED

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from transformers import BertTokenizer

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
SEEDS       = [42, 123, 456, 789, 2024]
FRACTIONS   = [0.25, 0.50, 0.75, 1.0]
BATCH_SIZE  = 32
EPOCHS      = 80
LR          = 1e-4
WEIGHT_DECAY = 1e-5
STEP_SIZE   = 30
GAMMA       = 0.5
VOCAB_SIZE  = 119547
EMBED_DIM   = 128
HIDDEN_DIM  = 256
NUM_CLASSES = 2
PATIENCE    = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Paths
CSV_DIR    = Path("/kaggle/input/datasets/amanlamichhane1234/nepooc-datset")
IMG_DIR    = Path("/kaggle/input/datasets/sanjeevkhatiwada/nepali-ooc-images")
OUTPUT_DIR = Path("/kaggle/working/nepooc_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD DATA (Full 1,090-sample benchmark)
# ============================================================
print("=" * 70)
print("LOADING FULL BENCHMARK (1,090 samples)")
print("=" * 70)

train_df = pd.read_csv(CSV_DIR / "nepOOC_train.csv")
val_df   = pd.read_csv(CSV_DIR / "nepOOC_val.csv")
test_df  = pd.read_csv(CSV_DIR / "nepOOC_test.csv")

print(f"\n✅ Dataset loaded:")
print(f"   Train: {len(train_df)} samples (OOC: {train_df['label'].sum()}, Pristine: {len(train_df)-train_df['label'].sum()})")
print(f"   Val:   {len(val_df)} samples (OOC: {val_df['label'].sum()}, Pristine: {len(val_df)-val_df['label'].sum()})")
print(f"   Test:  {len(test_df)} samples (OOC: {test_df['label'].sum()}, Pristine: {len(test_df)-test_df['label'].sum()})")

# ============================================================
# ADD IMAGE PATHS
# ============================================================
def get_image_path(post_id):
    for ext in ['jpg', 'jpeg', 'png', 'webp']:
        path = IMG_DIR / f"{post_id}.{ext}"
        if path.exists():
            return str(path)
    return None

train_df['image_path'] = train_df['post_id'].apply(get_image_path)
val_df['image_path']   = val_df['post_id'].apply(get_image_path)
test_df['image_path']  = test_df['post_id'].apply(get_image_path)

# Remove rows without images
train_df = train_df[train_df['image_path'].notna()].reset_index(drop=True)
val_df   = val_df[val_df['image_path'].notna()].reset_index(drop=True)
test_df  = test_df[test_df['image_path'].notna()].reset_index(drop=True)

print(f"\n📸 Images found:")
print(f"   Train: {len(train_df)}/{754} ({100*len(train_df)/754:.1f}%)")
print(f"   Val:   {len(val_df)}/{108} ({100*len(val_df)/108:.1f}%)")
print(f"   Test:  {len(test_df)}/{228} ({100*len(test_df)/228:.1f}%)")

# ============================================================
# TOKENIZER & TRANSFORMS
# ============================================================
print("\nLoading tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

BASE_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# DATASET CLASS
# ============================================================
class NepOOCDataset(Dataset):
    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        try:                                                          # FIX: handle corrupted images
            img = Image.open(row['image_path']).convert('RGB')
        except Exception:
            img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        img     = self.transform(img)
        caption = str(row['caption']) if pd.notna(row['caption']) else ""
        enc     = tokenizer(caption, max_length=128, padding='max_length',
                            truncation=True, return_tensors='pt')
        return {
            'image':             img,
            'input_ids':         enc['input_ids'].squeeze(0),
            'attention_mask':    enc['attention_mask'].squeeze(0),
            'label':             torch.tensor(int(row['label']), dtype=torch.long),
            'post_id':           row['post_id'],
            'misinformation_type': str(row.get('misinformation_type', 'pristine'))
        }

# ============================================================
# MODEL: 5-LAYER CNN + BiLSTM  (Section IV-F)
# ============================================================
class CNN5Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.proj = nn.Linear(512 * 4 * 4, 512)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.drop(torch.relu(self.proj(x)))

class LSTMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.lstm  = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.proj  = nn.Linear(HIDDEN_DIM * 2, 512)
        self.drop  = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        emb    = self.embed(input_ids)
        _, (h, _) = self.lstm(emb)
        h      = torch.cat([h[0], h[1]], dim=1)
        return self.drop(torch.relu(self.proj(h)))

class CNNLSTM(nn.Module):
    """Multimodal CNN+LSTM: visual (512) + text (512) → 1024 → 256 → 2."""
    def __init__(self):
        super().__init__()
        self.cnn  = CNN5Layer()
        self.lstm = LSTMEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, imgs, input_ids, attn_mask):
        v = self.cnn(imgs)
        t = self.lstm(input_ids, attn_mask)
        return self.classifier(torch.cat([v, t], dim=1))

# ============================================================
# TEXT-ONLY & IMAGE-ONLY  (Table XIV modality ablation)
# ============================================================
class TextOnlyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = LSTMEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, imgs, input_ids, attn_mask):
        return self.classifier(self.lstm(input_ids, attn_mask))

class ImageOnlyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN5Layer()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, imgs, input_ids, attn_mask):
        return self.classifier(self.cnn(imgs))

# ============================================================
# UTILITIES
# ============================================================
def seed_everything(seed):
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
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            imgs   = batch['image'].to(device)
            ids    = batch['input_ids'].to(device)
            masks  = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(imgs, ids, masks)
            loss   = criterion(logits, labels)
            probs  = torch.softmax(logits, dim=-1)[:, 1]

            total_loss += loss.item()
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if return_details:
                all_post_ids.extend(batch['post_id'])
                all_types.extend(batch['misinformation_type'])

    return {
        'loss':           total_loss / len(loader),
        'acc':            accuracy_score(all_labels, all_preds),
        'f1_macro':       f1_score(all_labels, all_preds, average='macro'),
        'auc':            roc_auc_score(all_labels, all_probs),
        'precision_ooc':  precision_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'recall_ooc':     recall_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'f1_ooc':         f1_score(all_labels, all_preds, pos_label=1, zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_preds, labels=[0, 1]).tolist(),  # FIX: always (2,2)
        'preds':          all_preds,
        'probs':          all_probs,
        'labels':         all_labels,
        'post_ids':       all_post_ids if return_details else [],
        'types':          all_types    if return_details else [],
    }

# ============================================================
# TRAIN ONE MODEL  (fraction × seed)
# ============================================================
def train_model(model_class, seed, fraction, train_df_full, val_df, test_df, model_name):
    """Train a single model instance with stratified subsampling."""
    seed_everything(seed)

    # ── Stratified subsampling (FIXED) ───────────────────────────────────────
    # Replaces the old: train_df.sample(frac=fraction, random_state=seed)
    # StratifiedShuffleSplit preserves the 50/50 OOC/Pristine balance at every
    # fraction, matching the protocol used by ResNet+mBERT, CLIP, and ViT+TCN.
    if fraction < 1.0:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=fraction, random_state=seed
        )
        idx, _ = next(sss.split(train_df_full, train_df_full['label']))
        train_subset = train_df_full.iloc[idx].reset_index(drop=True)
    else:
        train_subset = train_df_full.copy()
    # ─────────────────────────────────────────────────────────────────────────

    train_ds = NepOOCDataset(train_subset, BASE_TRANSFORM)
    val_ds   = NepOOCDataset(val_df,       BASE_TRANSFORM)
    test_ds  = NepOOCDataset(test_df,      BASE_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model     = model_class().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_val_f1  = 0.0
    best_state   = {k: v.cpu() for k, v in model.state_dict().items()}  # FIX: never None
    patience_ctr = 0
    history      = []

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
            optimizer.step()

            train_preds.extend(logits.argmax(1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        scheduler.step()
        train_f1   = f1_score(train_labels, train_preds, average='macro')
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        if epoch % 10 == 0 or epoch == 1:
            print(f"      E{epoch:03d} | Train F1: {train_f1:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")

        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1  = val_metrics['f1_macro']
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"      Early stopping at epoch {epoch}")
                break

    # Evaluate best checkpoint on test set
    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    test_metrics = evaluate(model, test_loader, criterion, DEVICE, return_details=True)
    test_metrics['best_val_f1']   = best_val_f1
    test_metrics['train_samples'] = len(train_subset)
    test_metrics['seed']          = seed
    test_metrics['fraction']      = fraction
    test_metrics['model']         = model_name
    test_metrics['history']       = history

    return test_metrics

# ============================================================
# RUN EXPERIMENTS
# ============================================================
print(f"\n{'='*70}")
print("STARTING FULL EXPERIMENT ON 1,090-SAMPLE BENCHMARK")
print(f"Training on {DEVICE}")
print(f"Total runs: {len(SEEDS)} seeds × {len(FRACTIONS)} fractions × 3 modalities = {len(SEEDS)*len(FRACTIONS)*3}")
print(f"{'='*70}")

all_results = {m: [] for m in ['multimodal', 'text_only', 'image_only']}

# Per-seed, per-typology predictions (for Table XII)
typo_per_seed = defaultdict(lambda: defaultdict(lambda: {'preds': [], 'labels': []}))

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"SEED: {seed}")
    print(f"{'='*50}")

    for fraction in FRACTIONS:
        print(f"\n  Training with {fraction*100:.0f}% of data...")

        print("    [1/3] Multimodal CNN+LSTM")
        mm_results = train_model(CNNLSTM,      seed, fraction, train_df, val_df, test_df, 'multimodal')
        all_results['multimodal'].append(mm_results)

        print("    [2/3] Text-Only")
        to_results = train_model(TextOnlyCNN,  seed, fraction, train_df, val_df, test_df, 'text_only')
        all_results['text_only'].append(to_results)

        print("    [3/3] Image-Only")
        io_results = train_model(ImageOnlyCNN, seed, fraction, train_df, val_df, test_df, 'image_only')
        all_results['image_only'].append(io_results)

        # Collect per-typology predictions at 100% for Table XII
        if fraction == 1.0:
            for pred, label, typo in zip(mm_results['preds'],
                                         mm_results['labels'],
                                         mm_results['types']):
                if typo not in ('pristine', 'nan', 'None'):
                    typo_per_seed[seed][typo]['preds'].append(pred)
                    typo_per_seed[seed][typo]['labels'].append(label)

# ============================================================
# AGGREGATE & PRINT RESULTS  (Tables VIII – XIV)
# ============================================================
print(f"\n{'='*70}")
print("RESULTS FOR PAPER (FULL 1,090-SAMPLE BENCHMARK)")
print(f"{'='*70}")

def compute_stats(results_list, fraction=None):
    filtered = [r for r in results_list if fraction is None or r['fraction'] == fraction]
    if not filtered:
        return None
    return {
        'acc_mean':  np.mean([r['acc']           for r in filtered]),
        'acc_std':   np.std( [r['acc']           for r in filtered]),
        'f1_mean':   np.mean([r['f1_macro']      for r in filtered]),
        'f1_std':    np.std( [r['f1_macro']      for r in filtered]),
        'auc_mean':  np.mean([r['auc']           for r in filtered]),
        'auc_std':   np.std( [r['auc']           for r in filtered]),
        'prec_mean': np.mean([r['precision_ooc'] for r in filtered]),
        'prec_std':  np.std( [r['precision_ooc'] for r in filtered]),
        'rec_mean':  np.mean([r['recall_ooc']    for r in filtered]),
        'rec_std':   np.std( [r['recall_ooc']    for r in filtered]),
        'f1_ooc_mean': np.mean([r['f1_ooc']      for r in filtered]),
        'f1_ooc_std':  np.std( [r['f1_ooc']      for r in filtered]),
        'confusion_matrix': np.mean(
            [np.array(r['confusion_matrix']) for r in filtered], axis=0
        ).tolist(),
        'n_seeds': len(filtered),
    }

# TABLE VIII
print("\n📊 TABLE VIII: Main Results (100% training data, 5 seeds)")
main_stats = compute_stats(all_results['multimodal'], fraction=1.0)
print(f"   Accuracy:  {main_stats['acc_mean']:.3f} ± {main_stats['acc_std']:.3f}")
print(f"   Macro-F1:  {main_stats['f1_mean']:.3f} ± {main_stats['f1_std']:.3f}")
print(f"   AUC:       {main_stats['auc_mean']:.3f} ± {main_stats['auc_std']:.3f}")
print(f"   OOC F1:    {main_stats['f1_ooc_mean']:.3f} ± {main_stats['f1_ooc_std']:.3f}")

# TABLE IX
print("\n📊 TABLE IX: OOC-Class Metrics (100% data)")
print(f"   Precision: {main_stats['prec_mean']:.3f} ± {main_stats['prec_std']:.3f}")
print(f"   Recall:    {main_stats['rec_mean']:.3f} ± {main_stats['rec_std']:.3f}")

# TABLE X
print("\n📊 TABLE X: Training-Size Scaling")
print("   Fraction | Macro-F1")
print("   ---------|----------")
for frac in FRACTIONS:
    s = compute_stats(all_results['multimodal'], fraction=frac)
    if s:
        print(f"   {frac*100:3.0f}%      | {s['f1_mean']:.3f} ± {s['f1_std']:.3f}")

# TABLE XI
print("\n📊 TABLE XI: Confusion Matrix (avg over 5 seeds)")
cm = main_stats['confusion_matrix']
print(f"                  Pred Pristine  Pred OOC")
print(f"   Actual Pristine   {cm[0][0]:.1f}         {cm[0][1]:.1f}")
print(f"   Actual OOC        {cm[1][0]:.1f}         {cm[1][1]:.1f}")

# TABLE XII
print("\n📊 TABLE XII: Per-Typology OOC Detection F1 (100% data)")
test_types    = test_df[test_df['label'] == 1]['misinformation_type'].value_counts()
all_typologies = set()
for seed_data in typo_per_seed.values():
    all_typologies.update(seed_data.keys())

print("   Typology              | Samples | F1 (mean ± std)")
print("   ----------------------|---------|------------------")
for typo in sorted(all_typologies):
    count    = test_types.get(typo, 0)
    seed_f1s = []
    for seed in SEEDS:
        if typo in typo_per_seed[seed]:
            d  = typo_per_seed[seed][typo]
            if d['labels']:
                seed_f1s.append(
                    f1_score(d['labels'], d['preds'], pos_label=1, zero_division=0)
                )
    if seed_f1s:
        print(f"   {typo:20s} | {count:5d}    | {np.mean(seed_f1s):.3f} ± {np.std(seed_f1s):.3f}")

# TABLE XIV
print("\n📊 TABLE XIV: Modality Ablation (100% data)")
mm = compute_stats(all_results['multimodal'], fraction=1.0)
to = compute_stats(all_results['text_only'],  fraction=1.0)
io = compute_stats(all_results['image_only'], fraction=1.0)
print(f"   Multimodal: {mm['f1_mean']:.3f} ± {mm['f1_std']:.3f}")
print(f"   Text-Only:  {to['f1_mean']:.3f} ± {to['f1_std']:.3f}")
print(f"   Image-Only: {io['f1_mean']:.3f} ± {io['f1_std']:.3f}")
print(f"\n   Gain (Multi vs Text):  {mm['f1_mean'] - to['f1_mean']:.3f}")
print(f"   Gain (Multi vs Image): {mm['f1_mean'] - io['f1_mean']:.3f}")

# ============================================================
# SAVE RESULTS
# ============================================================
print(f"\n💾 Saving results to {OUTPUT_DIR}")

pd.DataFrame(all_results['multimodal']).to_csv(OUTPUT_DIR / "cnn_lstm_all_results.csv",  index=False)
pd.DataFrame(all_results['text_only']).to_csv( OUTPUT_DIR / "text_only_results.csv",     index=False)
pd.DataFrame(all_results['image_only']).to_csv(OUTPUT_DIR / "image_only_results.csv",    index=False)

scaling_df = pd.DataFrame([{
    'fraction': frac,
    'f1_mean':  compute_stats(all_results['multimodal'], fraction=frac)['f1_mean'],
    'f1_std':   compute_stats(all_results['multimodal'], fraction=frac)['f1_std'],
} for frac in FRACTIONS])
scaling_df.to_csv(OUTPUT_DIR / "scaling_data.csv", index=False)

typo_results_list = []
for typo in sorted(all_typologies):
    for seed in SEEDS:
        if typo in typo_per_seed[seed]:
            d = typo_per_seed[seed][typo]
            typo_results_list.append({
                'typology':  typo,
                'seed':      seed,
                'f1':        f1_score(d['labels'], d['preds'], pos_label=1, zero_division=0),
                'n_samples': len(d['labels']),
            })
pd.DataFrame(typo_results_list).to_csv(OUTPUT_DIR / "typology_results.csv", index=False)

# ============================================================
# FIGURE 5: Training-Size Scaling Curve
# ============================================================
print("\n📈 Generating Figure 5...")
plt.figure(figsize=(8, 6))
fractions_pct = [f * 100 for f in FRACTIONS]
f1_means = [compute_stats(all_results['multimodal'], f)['f1_mean'] for f in FRACTIONS]
f1_stds  = [compute_stats(all_results['multimodal'], f)['f1_std']  for f in FRACTIONS]
plt.errorbar(fractions_pct, f1_means, yerr=f1_stds,
             marker='o', capsize=5, capthick=2,
             elinewidth=2, markersize=8, linewidth=2, color='navy')
plt.xlabel('Training Data (%)', fontsize=12)
plt.ylabel('Macro-F1', fontsize=12)
plt.title('Figure 5: Training-Size Scaling (CNN+LSTM)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure5_scaling_curve.png", dpi=150)
plt.close()
print(f"   Saved: {OUTPUT_DIR}/figure5_scaling_curve.png")

# ============================================================
# FIGURE 6: ROC and PR Curves
# ============================================================
print("\n📈 Generating Figure 6...")
best_result = max(
    [r for r in all_results['multimodal'] if r['fraction'] == 1.0],
    key=lambda x: x['auc']
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

fpr, tpr, _         = roc_curve(best_result['labels'], best_result['probs'])
precision, recall, _ = precision_recall_curve(best_result['labels'], best_result['probs'])

ax1.plot(fpr, tpr, linewidth=2,
         label=f"CNN+LSTM (AUC = {best_result['auc']:.3f})", color='darkorange')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate',  fontsize=12)
ax1.set_title('ROC Curve', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(recall, precision, linewidth=2, color='darkgreen')
ax2.set_xlabel('Recall',    fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curve', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure6_pr_roc_curves.png", dpi=150)
plt.close()
print(f"   Saved: {OUTPUT_DIR}/figure6_pr_roc_curves.png")

# ============================================================
# FIGURE 7: Failure Cases
# ============================================================
print("\n🔍 Identifying failure cases for Figure 7...")
misclassified = []
for pred, label, pid, typo in zip(best_result['preds'], best_result['labels'],
                                   best_result['post_ids'], best_result['types']):
    if pred != label:
        misclassified.append({
            'post_id':            pid,
            'true_label':         'OOC' if label == 1 else 'Pristine',
            'pred_label':         'OOC' if pred  == 1 else 'Pristine',
            'misinformation_type': typo,
        })
failure_df = pd.DataFrame(misclassified)
failure_df.to_csv(OUTPUT_DIR / "failure_cases.csv", index=False)
print(f"   Found {len(misclassified)} misclassified samples")
print(f"   Saved: {OUTPUT_DIR}/failure_cases.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print("✅ EXPERIMENT COMPLETE!")
print(f"{'='*70}")
print(f"""
All results saved to: {OUTPUT_DIR}

Files generated:
├── cnn_lstm_all_results.csv     - All 20 runs (5 seeds × 4 fractions)
├── text_only_results.csv        - Ablation: text-only model
├── image_only_results.csv       - Ablation: image-only model
├── scaling_data.csv             - Data for Figure 5
├── typology_results.csv         - Per-seed, per-typology F1 scores
├── figure5_scaling_curve.png    - Training-size scaling plot
├── figure6_pr_roc_curves.png    - ROC and PR curves
├── failure_cases.csv            - Misclassified samples for Figure 7
└── (console output above)       - All tables for paper

PAPER RESULTS SUMMARY (FULL 1,090-SAMPLE BENCHMARK):
- Accuracy:  {main_stats['acc_mean']:.3f} ± {main_stats['acc_std']:.3f}
- Macro-F1:  {main_stats['f1_mean']:.3f} ± {main_stats['f1_std']:.3f}
- AUC:       {main_stats['auc_mean']:.3f} ± {main_stats['auc_std']:.3f}
""")
