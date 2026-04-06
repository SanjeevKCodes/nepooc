"""
nepooc_05_vit_muril.py
ViT-B/16 (pretrained) + MuRIL + cross-attention fusion
MATCHES METHODOLOGY EXACTLY:
- ViT-B/16 pretrained on ImageNet-21k, 448x448 resolution
- MuRIL with LoRA (r=8, α=16)
- Fusion: Single-head cross-attention (ViT patches query MuRIL tokens)
- Classifier: 768 -> 256 -> 2
- Batch size: 16 per GPU (effective 32 across 2 GPUs)
- LR: 1e-4 (vision) / 2e-5 (LoRA), AdamW, weight_decay: 1e-4
- Cosine scheduler with 10% warmup
- NO data augmentation
- Correct class weights: 0.857/1.200
- NO label smoothing, NO gradient clipping, NO mixed precision
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

# ── Config (MATCHING PDF METHODOLOGY) ─────────────────────────────────────────
SEEDS       = [42, 123, 456, 789, 2024]
FRACTIONS   = [0.25, 0.50, 0.75, 1.0]
BATCH_SIZE  = 16                    # Per GPU (2 GPUs → effective 32)
GRAD_ACCUM_STEPS = 1                # No accumulation needed
EPOCHS      = 100                   # ✅ PDF Table VI: 100
LR_VISION   = 1e-4                  # ✅ PDF Table VI: 1e-4
LR_TEXT     = 2e-5                  # ✅ PDF Table VI: 2e-5
WEIGHT_DECAY = 1e-4                 # ✅ PDF Table VI: 1e-4
NUM_CLASSES = 2
PATIENCE    = 10
LORA_RANK   = 8                     # ✅ PDF Section IV-E
LORA_ALPHA  = 16                    # ✅ PDF Section IV-E
MURIL_ID    = "google/muril-base-cased"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 448                   # ✅ PDF Section IV-E
PATCH_SIZE  = 16                    # ✅ ViT-B/16
VIT_DIM     = 768                   # ✅ ViT-B/16 hidden dim

print(f"Device: {DEVICE}, GPUs: {torch.cuda.device_count()}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}, Patches: {IMG_SIZE//PATCH_SIZE}x{IMG_SIZE//PATCH_SIZE}")

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

# ✅ CORRECT class weights (PDF Section IV-A)
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

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("Loading MuRIL tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MURIL_ID)
print("MuRIL tokenizer ready ✅")

# ✅ NO DATA AUGMENTATION (PDF Section IV-B)
# Same transform for all splits
BASE_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

# ── Helper: Resize ViT positional embeddings ─────────────────────────────────
def resize_vit_positional_embeddings(model, new_img_size=448, patch_size=16):
    """Interpolate ViT positional embeddings for larger images"""
    if not hasattr(model, 'pos_embed'):
        return model
    
    old_pe = model.pos_embed.data
    old_hw = int((old_pe.shape[1] - 1) ** 0.5)
    new_hw = new_img_size // patch_size
    
    if new_hw == old_hw:
        return model
    
    print(f"Interpolating pos_embed: {old_hw}x{old_hw} -> {new_hw}x{new_hw}")
    
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

# ── ViT Encoder (448px, pretrained, NO freezing) ─────────────────────────────
class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # ✅ Pretrained ViT-B/16 from ImageNet-21k
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        # ✅ Resize positional embeddings for 448x448
        self.vit = resize_vit_positional_embeddings(self.vit, IMG_SIZE, PATCH_SIZE)
        # ✅ NO freezing (all layers trainable)
        # ✅ Keep 768 dim (NO projection to smaller dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        feat = self.vit(x)  # [B, 768]
        return self.drop(feat)  # Keep 768 dim

# ── MuRIL with LoRA ──────────────────────────────────────────────────────────
def apply_lora_to_muril(model):
    """Apply LoRA to MuRIL's attention layers (query and value)"""
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

print("Loading MuRIL model...")
muril_base = AutoModel.from_pretrained(MURIL_ID)
muril_base = apply_lora_to_muril(muril_base)
print("MuRIL + LoRA ready ✅")

class MuRILEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.muril = muril_base
        # ✅ NO projection (keep 768 dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        out = self.muril(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [B, 768]
        return self.drop(cls)  # Keep 768 dim

# ── Cross-Attention Fusion (ViT queries MuRIL, Single-head) ───────────────────
class CrossAttentionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # ✅ Single-head attention (h=1) per PDF Section IV-E
        # ✅ ViT patches query MuRIL tokens
        self.attn = nn.MultiheadAttention(VIT_DIM, 1, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(VIT_DIM)

    def forward(self, v_img, v_text):
        # ✅ CORRECT DIRECTION: ViT patches query MuRIL tokens
        q = v_img.unsqueeze(1)    # ViT as Query
        k = v_text.unsqueeze(1)   # MuRIL as Key
        v = v_text.unsqueeze(1)   # MuRIL as Value
        attn_out, _ = self.attn(q, k, v)
        return self.norm(attn_out.squeeze(1) + v_img)

# ── Complete Model ───────────────────────────────────────────────────────────
class ModelE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTEncoder()
        self.muril = MuRILEncoder()
        self.fusion = CrossAttentionFusion()
        # ✅ Classifier: 768 -> 256 -> 2
        self.classifier = nn.Sequential(
            nn.Linear(VIT_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, imgs, input_ids, attn_mask):
        v_img = self.vit(imgs)                    # [B, 768]
        v_text = self.muril(input_ids, attn_mask) # [B, 768]
        v_fused = self.fusion(v_img, v_text)      # [B, 768]
        return self.classifier(v_fused)

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

# ── Training function (MATCHING PDF METHODOLOGY) ──────────────────────────────
def run_vit_muril(seed, data_fraction):
    print(f"\n{'='*55}")
    print(f"ViT+MuRIL+LoRA+448px | Seed={seed} | Fraction={data_fraction}")
    print(f"{'='*55}")
    
    seed_everything(seed)
    device = DEVICE

    train_subset = train_df.sample(frac=data_fraction, random_state=seed).reset_index(drop=True)
    # ✅ Same transform for all splits (NO augmentation)
    train_ds = NepOOCDataset(train_subset, BASE_TRANSFORM)
    val_ds = NepOOCDataset(val_df, BASE_TRANSFORM)
    test_ds = NepOOCDataset(test_df, BASE_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ModelE().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        print(f"Effective batch size: {BATCH_SIZE} × {torch.cuda.device_count()} = {BATCH_SIZE * torch.cuda.device_count()}")

    # Extract LoRA parameters correctly
    def get_lora_params(m):
        if hasattr(m, 'module'):
            m = m.module
        return [p for p in m.muril.muril.parameters() if p.requires_grad]

    lora_params = get_lora_params(model)
    other_params = [p for p in model.parameters() if not any(p is lp for lp in lora_params)]

    print(f"LoRA trainable params: {sum(p.numel() for p in lora_params):,}")
    print(f"Other params: {sum(p.numel() for p in other_params):,}")

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR_TEXT},
        {"params": other_params, "lr": LR_VISION}
    ], weight_decay=WEIGHT_DECAY)

    # ✅ Cosine scheduler with 10% warmup (PDF Section IV-G)
    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(0.10 * total_steps)  # 10% warmup
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    weights = CLASS_WEIGHTS.to(device)
    
    # ✅ NO label smoothing (not in PDF methodology)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # ❌ NO gradient clipping, NO mixed precision

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
            # ❌ NO gradient clipping
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
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
    torch.save(best_state, f"results/model_ViT_MuRIL_seed{seed}_frac{data_fraction}.pth")
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