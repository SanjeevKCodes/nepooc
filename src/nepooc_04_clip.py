"""
nepooc_04_clip.py
CLIP ViT-B/32 fine-tuned (cosine similarity + learned head)
MATCHES METHODOLOGY EXACTLY:
- CLIP ViT-B/32 fine-tuned
- Class weights: 0.857/1.200 (inverse frequency)
- Concatenation: [v, t, cos_sim, v-t] -> 1537 dim
- MLP: 1537 -> 512 -> 256 -> 2
- Batch size: 32, LR: 2e-5, AdamW, weight_decay: 1e-4
- Cosine scheduler (NO warmup)
- NO label smoothing, NO gradient clipping, NO mixed precision
- NO different LR for head (same LR for all parameters)
"""

import os, subprocess, sys, json, time, random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── Install CLIP ──────────────────────────────────────────────────────────────
os.system("pip install -q git+https://github.com/openai/CLIP.git")
import clip

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

# ── Config (MATCHING PDF METHODOLOGY) ─────────────────────────────────────────
SEEDS       = [42, 123, 456, 789, 2024]
FRACTIONS   = [0.25, 0.50, 0.75, 1.0]
BATCH_SIZE  = 32                    # ✅ PDF Table VI: 32
EPOCHS      = 50                    # ✅ PDF Table VI: 50
LR          = 2e-5                  # ✅ PDF Table VI: 2e-5
WEIGHT_DECAY = 1e-4                 # ✅ PDF Table VI: 1e-4
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
        if p.exists():
            return str(p)
    return None

df['image_path'] = df['post_id'].apply(find_image)
df = df[df['image_path'].notna()].copy()
df['label_binary'] = (df['label_text'] == 'out_of_context').astype(int)

def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False

df['valid'] = df['image_path'].apply(is_valid_image)
df = df[df['valid']].copy()

# ✅ CORRECT class weights (PDF Section IV-A and IV-F)
n_total = len(df)
n_pristine = (df['label_binary'] == 0).sum()
n_ooc = (df['label_binary'] == 1).sum()
w_pristine = n_total / (2 * n_pristine)
w_ooc = n_total / (2 * n_ooc)
CLASS_WEIGHTS = torch.tensor([w_pristine, w_ooc], dtype=torch.float)
print(f"Class weights: Pristine={w_pristine:.3f}, OOC={w_ooc:.3f}")
print(f"Expected: 0.857, 1.200 — Actual: {w_pristine:.3f}, {w_ooc:.3f}")

train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df   = df[df['split'] == 'validation'].reset_index(drop=True)
test_df  = df[df['split'] == 'test'].reset_index(drop=True)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ── Load CLIP once ────────────────────────────────────────────────────────────
print("Loading CLIP ViT-B/32...")
clip_model_base, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
print("CLIP ready ✅")

# ── Dataset (CLIP native preprocessing - NO additional augmentation) ─────────
class CLIPDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        caption = str(row['caption'])[:77] if pd.notna(row['caption']) else ""
        try:
            img = Image.open(row['image_path']).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img_tensor = clip_preprocess(img)
        text_token = clip.tokenize([caption], truncate=True).squeeze(0)
        typology = str(row.get('misinformation_type', 'Pristine'))
        if typology == 'nan':
            typology = 'Pristine'
        return {
            'image': img_tensor,
            'text': text_token,
            'label': torch.tensor(int(row['label_binary']), dtype=torch.long),
            'typology': typology,
            'language': str(row.get('language', 'unknown'))
        }

# ── Model: CLIP fine-tuned (MATCHING PDF Section IV-F) ────────────────────────
class CLIPFineTuned(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        embed_dim = clip_model.visual.output_dim  # 512
        
        # ✅ MLP: 1537 -> 512 -> 256 -> 2 (matching PDF)
        # 1537 = 512(v) + 512(t) + 1(cos_sim) + 512(v-t)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1 + embed_dim, 512),  # 1537 -> 512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),                             # 512 -> 256
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)                      # 256 -> 2
        )

    def forward(self, imgs, text_tokens):
        img_feat = self.clip.encode_image(imgs).float()
        text_feat = self.clip.encode_text(text_tokens).float()
        
        img_norm = F.normalize(img_feat, dim=1)
        text_norm = F.normalize(text_feat, dim=1)
        cos_sim = (img_norm * text_norm).sum(dim=1, keepdim=True)
        diff = img_feat - text_feat
        
        # ✅ Concatenate: [v, t, cos_sim, v-t] -> 1537 dim
        fused = torch.cat([img_feat, text_feat, cos_sim, diff], dim=1)
        return self.classifier(fused)

# ── Train/Eval functions ──────────────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def evaluate_clip(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_probs, all_labels = 0, [], [], []
    all_typologies, all_languages = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            logits = model(imgs, texts)
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
    preds = np.array(res['preds'])
    labels = np.array(res['labels'])
    scores = {}
    for t in set(res['typologies']):
        idx = [i for i, x in enumerate(res['typologies']) if x == t]
        if idx:
            scores[t] = round(f1_score(labels[idx], preds[idx], average='binary', zero_division=0), 4)
    return scores

# ── Training runner (MATCHING PDF METHODOLOGY) ────────────────────────────────
def run_clip(seed, data_fraction):
    print(f"\n{'='*55}")
    print(f"CLIP | Seed={seed} | Fraction={data_fraction}")
    print(f"{'='*55}")
    seed_everything(seed)
    device = DEVICE

    train_subset = train_df.sample(frac=data_fraction, random_state=seed).reset_index(drop=True)
    train_ds = CLIPDataset(train_subset)
    val_ds = CLIPDataset(val_df)
    test_ds = CLIPDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = CLIPFineTuned(clip_model_base).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ✅ Same LR for all parameters (NO separate LR for head)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    weights = CLASS_WEIGHTS.to(device)
    
    # ✅ NO label smoothing (not in PDF methodology)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # ✅ Cosine scheduler (NO warmup, per Table VI)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # ❌ NO gradient clipping, NO mixed precision

    best_val_f1, best_state, patience_ctr, history = 0, None, 0, []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss, all_preds, all_labels = 0, [], []
        for batch in train_loader:
            imgs = batch['image'].to(device)
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(imgs, texts)
            loss = criterion(logits, labels)
            loss.backward()
            # ❌ NO gradient clipping
            optimizer.step()
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        scheduler.step()
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        val_m = evaluate_clip(model, val_loader, criterion, device)
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
    torch.save(best_state, f"results/model_CLIP_seed{seed}_frac{data_fraction}.pth")
    test_m = evaluate_clip(model, test_loader, criterion, device)
    print(f"\nTEST: Acc={test_m['acc']:.4f} F1={test_m['f1']:.4f} AUC={test_m['auc']:.4f}")

    result = {
        'model': 'CLIP',
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
    fname = f"results/results_CLIP_seed{seed}_frac{data_fraction}.json"
    with open(fname, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {fname}")
    return result

# ── Run all 20 ────────────────────────────────────────────────────────────────
all_results = []
for seed in SEEDS:
    for frac in FRACTIONS:
        result = run_clip(seed, frac)
        all_results.append(result)
        print(f"✅ Done: seed={seed} frac={frac} TestF1={result['test_f1']:.4f}")

print("\n" + "="*55)
print("ALL 20 CLIP RUNS COMPLETE")
print("="*55)
for r in all_results:
    print(f"seed={r['seed']} frac={r['fraction']} "
          f"Acc={r['test_acc']:.4f} F1={r['test_f1']:.4f} AUC={r['test_auc']:.4f}")