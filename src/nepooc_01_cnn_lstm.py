"""
nepooc_01_cnn_lstm.py
5-layer scratch CNN + BiLSTM baseline
Paste ALL of this as a single cell in Kaggle notebook nepooc_01_cnn_lstm
GPU T4x2, Internet ON, all 3 datasets attached
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

def push_results_to_github(token):
    os.system("git config user.email 'kaggle@nepooc.ai'")
    os.system("git config user.name 'Kaggle Runner'")
    os.system(f"git remote set-url origin https://{token}@github.com/SanjeevKCodes/nepooc.git")
    os.system("git add results/*.json")
    os.system("git commit -m 'Auto: CNN+LSTM correct results'")
    os.system("git push origin main")

# ── Config & Data ─────────────────────────────────────────────────────────────
SEEDS      = [42, 123, 456, 789, 2024]
FRACTIONS  = [0.25, 0.50, 0.75, 1.0]
BATCH_SIZE = 64
EPOCHS     = 80
LR         = 1e-4
VOCAB_SIZE = 30522
EMBED_DIM  = 128
HIDDEN_DIM = 256
NUM_CLASSES= 2
PATIENCE   = 10
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")

CSV_PATH = Path("/kaggle/input/datasets/sanjeevkhatiwada/nepali-ooc-misinformation/nepali_ooc_misinformation.csv")
IMG_DIR  = Path("/kaggle/input/datasets/sanjeevkhatiwada/nepali-ooc-images/images")

df = pd.read_csv(CSV_PATH)

def find_image(pid):
    for ext in ["jpg", "jpeg", "png", "webp"]:
        p = IMG_DIR / f"{pid}.{ext}"
        if p.exists():
            return str(p)
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

n_pristine   = (df['label_binary'] == 0).sum()
n_ooc        = (df['label_binary'] == 1).sum()
CLASS_WEIGHTS = torch.tensor([1.0, n_pristine / n_ooc], dtype=torch.float)
print(f"Class weights: {CLASS_WEIGHTS}")

train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df   = df[df['split'] == 'validation'].reset_index(drop=True)
test_df  = df[df['split'] == 'test'].reset_index(drop=True)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ── Tokenizer & Transforms ────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
print("Tokenizer ready ✅")

TRAIN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.3),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
EVAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class NepOOCDataset(Dataset):
    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
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
            'image':          img,
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(int(row['label_binary']), dtype=torch.long),
            'typology':       typology,
            'language':       str(row.get('language', 'unknown'))
        }

# ── Model: 5-layer CNN + BiLSTM ───────────────────────────────────────────────
class CNN5Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
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
        emb = self.embed(input_ids)
        _, (h, _) = self.lstm(emb)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.drop(torch.relu(self.proj(h)))

class ModelA(nn.Module):
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

print("ModelA (CNN+LSTM) ready ✅")

# ── Train/Eval functions ──────────────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED']       = str(seed)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_probs, all_labels = 0, [], [], []
    all_typologies, all_languages = [], []
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
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_typologies.extend(batch['typology'])
            all_languages.extend(batch['language'])
    return {
        'loss': total_loss / len(loader),
        'acc':  accuracy_score(all_labels, all_preds),
        'f1':   f1_score(all_labels, all_preds, average='macro'),
        'auc':  roc_auc_score(all_labels, all_probs),
        'preds':      [int(p) for p in all_preds],
        'probs':      [float(p) for p in all_probs],
        'labels':     [int(l) for l in all_labels],
        'typologies': all_typologies,
        'languages':  all_languages
    }

def get_typology_f1(res):
    preds = np.array(res['preds']); labels = np.array(res['labels'])
    scores = {}
    for t in set(res['typologies']):
        idx = [i for i, x in enumerate(res['typologies']) if x == t]
        if idx:
            scores[t] = round(f1_score(
                labels[idx], preds[idx], average='binary', zero_division=0), 4)
    return scores

# ── Training runner ───────────────────────────────────────────────────────────
def run_cnn_lstm(seed, data_fraction):
    print(f"\n{'='*55}")
    print(f"CNN+LSTM | Seed={seed} | Fraction={data_fraction}")
    print(f"{'='*55}")
    seed_everything(seed)
    device = DEVICE

    train_subset = train_df.sample(frac=data_fraction, random_state=seed).reset_index(drop=True)
    train_ds = NepOOCDataset(train_subset, TRAIN_TRANSFORM)
    val_ds   = NepOOCDataset(val_df,       EVAL_TRANSFORM)
    test_ds  = NepOOCDataset(test_df,      EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model     = ModelA().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    weights   = CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler('cuda')

    best_val_f1, best_state, patience_ctr, history = 0, None, 0, []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss, all_preds, all_labels = 0, [], []
        for batch in train_loader:
            imgs   = batch['image'].to(device)
            ids    = batch['input_ids'].to(device)
            masks  = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(imgs, ids, masks)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item()
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        scheduler.step()

        train_f1 = f1_score(all_labels, all_preds, average='macro')
        val_m    = evaluate(model, val_loader, criterion, device)
        elapsed  = time.time() - t0
        history.append({'epoch': epoch, 'train_f1': round(train_f1, 4),
                        'val_f1': round(val_m['f1'], 4), 'val_auc': round(val_m['auc'], 4),
                        'time_s': round(elapsed, 1)})
        print(f"E{epoch:02d} TrainF1={train_f1:.4f} ValF1={val_m['f1']:.4f} "
              f"ValAUC={val_m['auc']:.4f} ({elapsed:.0f}s)")

        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']; best_state = {k: v.cpu() for k, v in model.state_dict().items()}; patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}"); break

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, criterion, device)
    print(f"\nTEST: Acc={test_m['acc']:.4f} F1={test_m['f1']:.4f} AUC={test_m['auc']:.4f}")

    result = {
        'model': 'CNN_LSTM', 'seed': seed, 'fraction': data_fraction, 'split_type': 'random',
        'best_val_f1': round(best_val_f1, 4), 'train_history': history,
        'test_acc': round(test_m['acc'], 4), 'test_f1': round(test_m['f1'], 4),
        'test_auc': round(test_m['auc'], 4), 'test_preds': test_m['preds'],
        'test_probs': test_m['probs'], 'test_labels': test_m['labels'],
        'typology_f1': get_typology_f1(test_m)
    }
    fname = f"results/results_CNN_LSTM_seed{seed}_frac{data_fraction}.json"
    with open(fname, 'w') as f: json.dump(result, f, indent=2)
    print(f"Saved: {fname}")
    return result

# ── Run all 20 ────────────────────────────────────────────────────────────────
all_results = []
for seed in SEEDS:
    for frac in FRACTIONS:
        result = run_cnn_lstm(seed, frac)
        all_results.append(result)
        print(f"✅ Done: seed={seed} frac={frac} TestF1={result['test_f1']:.4f}")

print("\n" + "="*55)
print("ALL 20 CNN+LSTM RUNS COMPLETE")
print("="*55)
for r in all_results:
    print(f"seed={r['seed']} frac={r['fraction']} "
          f"Acc={r['test_acc']:.4f} F1={r['test_f1']:.4f} AUC={r['test_auc']:.4f}")

# TOKEN = "your-new-token-here"
# push_results_to_github(TOKEN)
