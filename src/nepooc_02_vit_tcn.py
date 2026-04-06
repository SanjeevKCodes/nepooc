"""
nepooc_02_vit_tcn.py
ViT-B/16 (pretrained) + TCN + cross-attention fusion
MATCHES METHODOLOGY EXACTLY:
- ViT-B/16 pretrained on ImageNet-21k
- Patch size: 16x16
- 12 transformer layers, 12 heads, 768 dim
- TCN: 3 blocks, dilations [1,2,4], kernel=3, hidden=256
- Fusion: Single-head cross-attention (text Q, image K,V)
- Classifier: 768->256->2
- Batch size: 32, LR: 5e-5, AdamW, weight_decay: 1e-4
- Cosine scheduler (no warmup)
- NO data augmentation
- Correct class weights: 0.857/1.200
"""

import os, subprocess, sys, json, time, random, math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Import pretrained ViT
import timm

# ── GitHub setup ──────────────────────────────────────────────────────────────
GITHUB_REPO = "https://github.com/SanjeevKCodes/nepooc.git"
if not os.path.exists("/kaggle/working/nepooc"):
    subprocess.run(["git", "clone", GITHUB_REPO, "/kaggle/working/nepooc"])
else:
    subprocess.run(["git", "-C", "/kaggle/working/nepooc", "pull"])

os.chdir("/kaggle/working/nepooc")
sys.path.insert(0, "/kaggle/working/nepooc")
os.makedirs("results", exist_ok=True)
print("Repo ready:", os.listdir("."))

# ── Config (MATCHING METHODOLOGY) ─────────────────────────────────────────────
SEEDS       = [42, 123, 456, 789, 2024]
FRACTIONS   = [0.25, 0.50, 0.75, 1.0]
BATCH_SIZE  = 32                    # ✅ Table VI
EPOCHS      = 100                   # ✅ Table VI
LR          = 5e-5                  # ✅ Table VI: 5e-5
WEIGHT_DECAY = 1e-4                 # ✅ Table VI
IMG_SIZE    = 224                   # ✅ ViT-B/16 default
PATCH_SIZE  = 16                    # ✅ ViT-B/16
VIT_DIM     = 768                   # ✅ ViT-B/16
VIT_HEADS   = 12                    # ✅ ViT-B/16
VIT_LAYERS  = 12                    # ✅ ViT-B/16
TCN_HIDDEN  = 256                   # ✅ Methodology
TCN_KERNEL  = 3
TCN_DILATIONS = [1, 2, 4]
NUM_CLASSES = 2
PATIENCE    = 10
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")

# ── Data loading ──────────────────────────────────────────────────────────────
CSV_PATH = Path("/kaggle/input/datasets/sanjeevkhatiwada/nepali-ooc-misinformation/nepali_ooc_misinformation.csv")
IMG_DIR  = Path("/kaggle/input/datasets/sanjeevkhatiwada/nepali-ooc-images/images")

df = pd.read_csv(CSV_PATH)

def find_image(pid):
    for ext in ["jpg", "jpeg", "png", "webp"]:
        p = IMG_DIR / f"{pid}.{ext}"
        if p.exists(): return str(p)
    return None

df['image_path']   = df['post_id'].apply(find_image)
df = df[df['image_path'].notna()].copy()
df['label_binary'] = (df['label_text'] == 'out_of_context').astype(int)

def is_valid_image(path):
    try:
        img = Image.open(path); img.verify(); return True
    except: return False

df['valid'] = df['image_path'].apply(is_valid_image)
df = df[df['valid']].copy()

# ✅ Correct class weights (inverse frequency)
n_total = len(df)
n_pristine = (df['label_binary'] == 0).sum()
n_ooc = (df['label_binary'] == 1).sum()
w_pristine = n_total / (2 * n_pristine)
w_ooc = n_total / (2 * n_ooc)
CLASS_WEIGHTS = torch.tensor([w_pristine, w_ooc], dtype=torch.float)
print(f"Class weights: Pristine={w_pristine:.3f}, OOC={w_ooc:.3f}")

train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df   = df[df['split'] == 'validation'].reset_index(drop=True)
test_df  = df[df['split'] == 'test'].reset_index(drop=True)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ── Tokenizer & Dataset ───────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
print("Tokenizer ready ✅")

# ✅ NO DATA AUGMENTATION (Methodology Section IV-B)
BASE_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class NepOOCDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        img     = Image.open(row['image_path']).convert('RGB')
        img     = self.transform(img)
        caption = str(row['caption']) if pd.notna(row['caption']) else ""
        enc     = tokenizer(caption, max_length=128, padding='max_length',
                            truncation=True, return_tensors='pt')
        typology = str(row.get('misinformation_type', 'Pristine'))
        if typology == 'nan': typology = 'Pristine'
        return {
            'image': img,
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(int(row['label_binary']), dtype=torch.long),
            'typology': typology,
            'language': str(row.get('language', 'unknown'))
        }

# ── Model: Pretrained ViT + TCN + Cross-Attention ─────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pad = pad

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.pad] if self.pad > 0 else out
        out = self.drop(self.act(self.norm(out)))
        return out + self.res(x)

class TCNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(30522, 128, padding_idx=0)
        self.tcn = nn.Sequential(
            TCNBlock(128, TCN_HIDDEN, TCN_KERNEL, dilation=TCN_DILATIONS[0]),
            TCNBlock(TCN_HIDDEN, TCN_HIDDEN, TCN_KERNEL, dilation=TCN_DILATIONS[1]),
            TCNBlock(TCN_HIDDEN, TCN_HIDDEN, TCN_KERNEL, dilation=TCN_DILATIONS[2]),
        )
        # ✅ Project to VIT_DIM (768) to match ViT dimension
        self.proj = nn.Linear(TCN_HIDDEN, VIT_DIM)
        self.drop = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids).transpose(1, 2)
        x = self.tcn(x)
        mask = attention_mask.unsqueeze(1).float()
        x = (x * mask).sum(2) / mask.sum(2).clamp(min=1)
        return self.drop(torch.relu(self.proj(x)))

class CrossAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # ✅ Single-head attention (h=1) per methodology
        self.attn = nn.MultiheadAttention(VIT_DIM, 1, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(VIT_DIM)

    def forward(self, v_img, v_text):
        # Text as Query, Image as Key/Value (per methodology)
        q = v_text.unsqueeze(1)  # [B, 1, D]
        k = v_img.unsqueeze(1)   # [B, 1, D]
        v = v_img.unsqueeze(1)   # [B, 1, D]
        fused, _ = self.attn(q, k, v)
        return self.norm(fused.squeeze(1) + v_text)

class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        # ✅ Pretrained ViT-B/16 from ImageNet-21k
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        # Freeze ViT? Methodology doesn't specify. Keep trainable for fine-tuning.
        self.tcn = TCNEncoder()
        self.fusion = CrossAttentionFusion()
        # ✅ Classifier: VIT_DIM (768) -> 256 -> 2
        self.classifier = nn.Sequential(
            nn.Linear(VIT_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, imgs, input_ids, attn_mask):
        v_img = self.vit(imgs)      # [B, 768] from ViT
        v_text = self.tcn(input_ids, attn_mask)  # [B, 768]
        v_fused = self.fusion(v_img, v_text)
        return self.classifier(v_fused)

print("ModelB (ViT+TCN with pretrained ViT) ready ✅")

# ── Train/Eval functions ──────────────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_probs, all_labels = 0, [], [], []
    all_typologies, all_languages = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(imgs, ids, masks)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_typologies.extend(batch['typology'])
            all_languages.extend(batch['language'])
    return {
        'loss': total_loss / len(loader),
        'acc': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'auc': roc_auc_score(all_labels, all_probs),
        'preds': [int(p) for p in all_preds],
        'probs': [float(p) for p in all_probs],
        'labels': [int(l) for l in all_labels],
        'typologies': all_typologies,
        'languages': all_languages
    }

def get_typology_f1(res):
    preds = np.array(res['preds']); labels = np.array(res['labels'])
    scores = {}
    for t in set(res['typologies']):
        idx = [i for i, x in enumerate(res['typologies']) if x == t]
        if idx:
            scores[t] = round(f1_score(labels[idx], preds[idx], average='binary', zero_division=0), 4)
    return scores

def run_vit_tcn(seed, data_fraction):
    print(f"\n{'='*55}")
    print(f"ViT+TCN | Seed={seed} | Fraction={data_fraction}")
    print(f"{'='*55}")
    seed_everything(seed)
    device = DEVICE

    train_subset = train_df.sample(frac=data_fraction, random_state=seed).reset_index(drop=True)
    # ✅ Same transform for all splits (no augmentation)
    train_ds = NepOOCDataset(train_subset, BASE_TRANSFORM)
    val_ds   = NepOOCDataset(val_df, BASE_TRANSFORM)
    test_ds  = NepOOCDataset(test_df, BASE_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ModelB().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    weights = CLASS_WEIGHTS.to(device)
    # ✅ No label smoothing (not in methodology)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # ✅ Cosine scheduler with NO warmup (per Table VI)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # ❌ No gradient clipping, no mixed precision (remove for methodology purity)

    best_val_f1, best_state, patience_ctr, history = 0, None, 0, []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss, all_preds, all_labels = 0, [], []
        for batch in train_loader:
            imgs = batch['image'].to(device)
            ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(imgs, ids, masks)
            loss = criterion(logits, labels)
            loss.backward()
            # ❌ No gradient clipping
            optimizer.step()
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        scheduler.step()
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        val_m = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        history.append({
            'epoch': epoch,
            'train_f1': round(train_f1, 4),
            'val_f1': round(val_m['f1'], 4),
            'val_auc': round(val_m['auc'], 4),
            'time_s': round(elapsed, 1)
        })
        print(f"E{epoch:02d} TrainF1={train_f1:.4f} ValF1={val_m['f1']:.4f} ValAUC={val_m['auc']:.4f} ({elapsed:.0f}s)")

        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, f"results/model_ViT_TCN_seed{seed}_frac{data_fraction}.pth")
    test_m = evaluate(model, test_loader, criterion, device)
    print(f"\nTEST: Acc={test_m['acc']:.4f} F1={test_m['f1']:.4f} AUC={test_m['auc']:.4f}")

    result = {
        'model': 'ViT_TCN',
        'seed': seed,
        'fraction': data_fraction,
        'split_type': 'random',
        'best_val_f1': round(best_val_f1, 4),
        'train_history': history,
        'test_acc': round(test_m['acc'], 4),
        'test_f1': round(test_m['f1'], 4),
        'test_auc': round(test_m['auc'], 4),
        'test_preds': test_m['preds'],
        'test_probs': test_m['probs'],
        'test_labels': test_m['labels'],
        'typology_f1': get_typology_f1(test_m)
    }
    fname = f"results/results_ViT_TCN_seed{seed}_frac{data_fraction}.json"
    with open(fname, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {fname}")
    return result

# ── Run all 20 ────────────────────────────────────────────────────────────────
all_results = []
for seed in SEEDS:
    for frac in FRACTIONS:
        result = run_vit_tcn(seed, frac)
        all_results.append(result)
        print(f"✅ Done: seed={seed} frac={frac} TestF1={result['test_f1']:.4f}")

print("\n" + "="*55)
print("ALL 20 ViT+TCN RUNS COMPLETE")
print("="*55)
for r in all_results:
    print(f"seed={r['seed']} frac={r['fraction']} Acc={r['test_acc']:.4f} F1={r['test_f1']:.4f} AUC={r['test_auc']:.4f}")