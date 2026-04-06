"""
nepooc_05_vit_muril.py
ViT-B/16 (pretrained timm) + MuRIL + cross-attention fusion
UPDATED: 448px image size + LoRA on MuRIL
Paste ALL of this as a single cell in Kaggle notebook nepooc_05_vit_muril
GPU T4x2, Internet ON, all 3 datasets attached
"""

import os
import subprocess
import sys
import json
import time
import random
import math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── Install dependencies ──────────────────────────────────────────────────────
os.system("pip install -q timm==0.9.16")
os.system("pip install -q peft")
import timm
from peft import get_peft_model, LoraConfig, TaskType

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
    os.system("git commit -m 'Auto: ViT+MuRIL correct results'")
    os.system("git push origin main")

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS      = [42, 123, 456, 789, 2024]
FRACTIONS  = [0.25, 0.50, 0.75, 1.0]
BATCH_SIZE = 8                    # Reduced to 8 for 448px images on T4
GRAD_ACCUM_STEPS = 2              # Effective batch size = 16
EPOCHS     = 100
LR_VISION  = 1e-4
LR_TEXT    = 2e-5
FUSE_DIM   = 512
NUM_CLASSES= 2
PATIENCE   = 10
LORA_RANK  = 8
LORA_ALPHA = 16
MURIL_ID   = "google/muril-base-cased"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 448
PATCH_SIZE = 16

print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}, Patches: {IMG_SIZE//PATCH_SIZE}x{IMG_SIZE//PATCH_SIZE}")

# ── Load data ─────────────────────────────────────────────────────────────────
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

n_pristine = (df['label_binary'] == 0).sum()
n_ooc = (df['label_binary'] == 1).sum()
CLASS_WEIGHTS = torch.tensor([1.0, n_pristine / n_ooc], dtype=torch.float)
print(f"Class weights: {CLASS_WEIGHTS}")

train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df = df[df['split'] == 'validation'].reset_index(drop=True)
test_df = df[df['split'] == 'test'].reset_index(drop=True)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("Loading MuRIL tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MURIL_ID)
print("MuRIL tokenizer ready ✅")

# ── Transforms for 448px images ───────────────────────────────────────────────
TRAIN_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.3),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

EVAL_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Dataset class ─────────────────────────────────────────────────────────────
class NepOOCDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        img = self.transform(img)
        caption = str(row['caption']) if pd.notna(row['caption']) else ""
        enc = tokenizer(caption, max_length=128, padding='max_length',
                        truncation=True, return_tensors='pt')
        typology = str(row.get('misinformation_type', 'Pristine'))
        if typology == 'nan':
            typology = 'Pristine'
        return {
            'image': img,
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(int(row['label_binary']), dtype=torch.long),
            'typology': typology,
            'language': str(row.get('language', 'unknown'))
        }

# ── ViT with 448px support (FIXED) ───────────────────────────────────────────
def resize_vit_positional_embeddings(model, new_img_size=448, patch_size=16):
    """Properly interpolate ViT positional embeddings for larger images"""
    if not hasattr(model, 'pos_embed'):
        print("Warning: No pos_embed found, skipping interpolation")
        return model
    
    old_pe = model.pos_embed.data
    old_hw = int((old_pe.shape[1] - 1) ** 0.5)
    new_hw = new_img_size // patch_size
    
    if new_hw == old_hw:
        print(f"Positional embeddings already match {new_hw}x{new_hw}")
        return model
    
    print(f"Interpolating: {old_hw}x{old_hw} -> {new_hw}x{new_hw}")
    
    with torch.no_grad():
        cls_token = old_pe[:, 0:1, :]
        pos_embed = old_pe[:, 1:, :]
        
        pos_embed = pos_embed.reshape(1, old_hw, old_hw, -1).permute(0, 3, 1, 2)
        pos_embed = torch.nn.functional.interpolate(
            pos_embed, size=(new_hw, new_hw), mode='bicubic', align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, -1)
        
        new_pe = torch.cat([cls_token, pos_embed], dim=1)
        model.pos_embed = nn.Parameter(new_pe)
    
    return model

print("Loading ViT-B/16 pretrained...")
vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
print(f"Original pos_embed shape: {vit_base.pos_embed.shape}")

vit_base = resize_vit_positional_embeddings(vit_base, IMG_SIZE, PATCH_SIZE)
print(f"New pos_embed shape: {vit_base.pos_embed.shape}")

# Test ViT forward pass
test_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
try:
    test_output = vit_base(test_input)
    print(f"✅ ViT forward pass successful, output shape: {test_output.shape}")
except Exception as e:
    print(f"❌ ViT forward pass failed: {e}")
    raise

print("ViT ready ✅ with 448px support")

# ── MuRIL with LoRA (FIXED) ───────────────────────────────────────────────────
print("Loading MuRIL model...")
muril_base = AutoModel.from_pretrained(MURIL_ID)
print("MuRIL ready ✅")

def apply_lora_to_muril(model):
    """Apply LoRA to MuRIL's attention layers"""
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied to MuRIL:")
    print(f"  - Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}% of total)")
    return model

muril_base = apply_lora_to_muril(muril_base)
print("MuRIL + LoRA ready ✅")

# ── Model components ──────────────────────────────────────────────────────────
# In nepooc_05_vit_muril.py, replace the ViTEncoder class with this:

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # EXPLICITLY load the model again INSIDE the class to ensure weights are used
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        # Resize positional embeddings for 448x448
        self.vit = resize_vit_positional_embeddings(self.vit, IMG_SIZE, PATCH_SIZE)
        
        # Freeze the early layers of ViT to prevent overfitting on small dataset
        for name, param in self.vit.named_parameters():
            if 'blocks.0' in name or 'blocks.1' in name or 'blocks.2' in name:
                param.requires_grad = False
        
        self.proj = nn.Linear(768, FUSE_DIM)
        self.drop = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(FUSE_DIM)

    def forward(self, x):
        feat = self.vit(x)
        return self.norm(self.drop(torch.relu(self.proj(feat))))

class CrossAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(FUSE_DIM, 8, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(FUSE_DIM)
        self.ff = nn.Sequential(
            nn.Linear(FUSE_DIM, FUSE_DIM * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(FUSE_DIM * 2, FUSE_DIM)
        )
        self.norm2 = nn.LayerNorm(FUSE_DIM)

    def forward(self, v_img, v_text):
        q = v_text.unsqueeze(1)
        k = v_img.unsqueeze(1)
        v = v_img.unsqueeze(1)
        attn_out, _ = self.attn(q, k, v)
        x = self.norm(attn_out.squeeze(1) + v_text)
        x = self.norm2(x + self.ff(x))
        return x

class ModelE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTEncoder()
        self.muril = MuRILEncoder()
        self.fusion = CrossAttentionFusion()
        self.classifier = nn.Sequential(
            nn.Linear(FUSE_DIM, 256), nn.GELU(), nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, imgs, input_ids, attn_mask):
        v_img = self.vit(imgs)
        v_text = self.muril(input_ids, attn_mask)
        v_f = self.fusion(v_img, v_text)
        return self.classifier(v_f)

print("ModelE (ViT+MuRIL+LoRA+448px) ready ✅")

# ── Helper functions ──────────────────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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
    preds = np.array(res['preds'])
    labels = np.array(res['labels'])
    scores = {}
    for t in set(res['typologies']):
        idx = [i for i, x in enumerate(res['typologies']) if x == t]
        if idx:
            scores[t] = round(f1_score(labels[idx], preds[idx], average='binary', zero_division=0), 4)
    return scores

# ── Training function ─────────────────────────────────────────────────────────
def run_vit_muril(seed, data_fraction):
    print(f"\n{'='*55}")
    print(f"ViT+MuRIL+LoRA+448px | Seed={seed} | Fraction={data_fraction}")
    print(f"{'='*55}")
    
    seed_everything(seed)
    device = DEVICE

    train_subset = train_df.sample(frac=data_fraction, random_state=seed).reset_index(drop=True)
    train_ds = NepOOCDataset(train_subset, TRAIN_TRANSFORM)
    val_ds = NepOOCDataset(val_df, EVAL_TRANSFORM)
    test_ds = NepOOCDataset(test_df, EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ModelE().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")

    # Extract LoRA parameters correctly
    def get_lora_params(m):
        if hasattr(m, 'module'):
            m = m.module
        if hasattr(m.muril, 'muril'):
            return [p for p in m.muril.muril.parameters() if p.requires_grad]
        return [p for p in m.muril.parameters() if p.requires_grad]

    lora_params = get_lora_params(model)
    other_params = [p for p in model.parameters() if not any(p is lp for lp in lora_params)]

    print(f"LoRA trainable params: {sum(p.numel() for p in lora_params):,}")
    print(f"Other params: {sum(p.numel() for p in other_params):,}")

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_TEXT},
        {"params": other_params, "lr": LR_VISION}
    ], weight_decay=1e-4)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    weights = CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    best_val_f1, best_state, patience_ctr, history, step = 0, None, 0, [], 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss, all_preds, all_labels = 0, [], []
        
        for i, batch in enumerate(train_loader):
            imgs = batch['image'].to(device)
            ids = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            with torch.amp.autocast('cuda'):
                logits = model(imgs, ids, masks)
                loss = criterion(logits, labels)
                loss = loss / GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
            
            total_loss += loss.item() * GRAD_ACCUM_STEPS
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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
    test_m = evaluate(model, test_loader, criterion, device)
    print(f"\nTEST: Acc={test_m['acc']:.4f} F1={test_m['f1']:.4f} AUC={test_m['auc']:.4f}")

    result = {
        'model': 'ViT_MuRIL_LoRA_448px',
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
    
    fname = f"results/results_ViT_MuRIL_LoRA_448px_seed{seed}_frac{data_fraction}.json"
    with open(fname, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {fname}")
    return result

# ── Run all 20 experiments ────────────────────────────────────────────────────
print("\n" + "="*55)
print("STARTING ALL 20 EXPERIMENTS")
print("="*55)

all_results = []
for seed in SEEDS:
    for frac in FRACTIONS:
        result = run_vit_muril(seed, frac)
        all_results.append(result)
        print(f"✅ Done: seed={seed} frac={frac} TestF1={result['test_f1']:.4f}")

print("\n" + "="*55)
print("ALL 20 ViT+MuRIL+LoRA+448px RUNS COMPLETE")
print("="*55)
for r in all_results:
    print(f"seed={r['seed']} frac={r['fraction']} Acc={r['test_acc']:.4f} F1={r['test_f1']:.4f} AUC={r['test_auc']:.4f}")

# Uncomment to push results to GitHub
# TOKEN = "your-github-token-here"
# push_results_to_github(TOKEN)