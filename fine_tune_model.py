# ====================== IMPORTS ======================
import os, copy, random, platform, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models
from torchvision.transforms import v2 as T
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import imagehash

# ====================== CONFIG ======================
DATA_DIR = "ppe_dataset"
DEDUPED_DATA_DIR = "ppe_dataset_deduped"

NUM_EPOCHS = 300
BATCH_SIZE = 12
BASE_LR = 1e-5
HEAD_LR = BASE_LR * 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EARLY_STOPPING_PATIENCE = 30
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.35
NUM_WORKERS = 0 if platform.system() == "Windows" else 4
RANDOM_SEED = 42
INPUT_SIZE = 224
ENABLE_DEDUPLICATION = True
SIMILARITY_THRESHOLD = 0.90
USE_TTA_IN_VALIDATION = True
TTA_AUGMENTS = 5
MODEL_CHOICE = 'swin'  # Swin-T
GRAPH_SAVE_PATH = "training_curve.png"

print(f"--- PPE Classification (Swin-T) ---")
print(f"Device: {DEVICE}, Input: {INPUT_SIZE}x{INPUT_SIZE}, Model: {MODEL_CHOICE}")

# ====================== REPRODUCIBILITY ======================
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed()

# ====================== DEDUPLICATION ======================
def deduplicate_frames(source_dir, dest_dir, similarity_threshold=SIMILARITY_THRESHOLD):
    if os.path.exists(dest_dir):
        print(f"✅ Using existing deduplicated dataset at: {dest_dir}")
        return dest_dir
    os.makedirs(dest_dir, exist_ok=True)
    hash_diff_threshold = int((1 - similarity_threshold) * 64)
    for split in ['train','val']:
        split_src, split_dst = os.path.join(source_dir, split), os.path.join(dest_dir, split)
        if not os.path.exists(split_src): continue
        os.makedirs(split_dst, exist_ok=True)
        for class_name in os.listdir(split_src):
            class_src, class_dst = os.path.join(split_src, class_name), os.path.join(split_dst, class_name)
            if not os.path.isdir(class_src): continue
            os.makedirs(class_dst, exist_ok=True)
            print(f"Processing {split}/{class_name}...")
            hashes, kept_count, removed_count = {},0,0
            for img_file in [f for f in os.listdir(class_src) if f.lower().endswith(('.jpg','.jpeg','.png'))]:
                try:
                    img = Image.open(os.path.join(class_src,img_file))
                    img_hash = imagehash.phash(img)
                    if not any(img_hash - h < hash_diff_threshold for h in hashes):
                        shutil.copy2(os.path.join(class_src,img_file), class_dst)
                        hashes[img_hash] = img_file; kept_count += 1
                    else: removed_count +=1
                except Exception as e: print(f"  Error {img_file}: {e}")
            print(f"  Kept: {kept_count}, Removed: {removed_count}")
    print(f"✅ Deduplication complete. Using data from: {dest_dir}")
    return dest_dir

working_data_dir = deduplicate_frames(DATA_DIR, DEDUPED_DATA_DIR, SIMILARITY_THRESHOLD) if ENABLE_DEDUPLICATION else DATA_DIR

# ====================== AUGMENTATIONS ======================
data_transforms = {
    'train': A.Compose([
        A.Resize(int(INPUT_SIZE*1.15), int(INPUT_SIZE*1.15)),
        A.RandomCrop(INPUT_SIZE, INPUT_SIZE),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
        A.CoarseDropout(max_holes=8, max_height=int(INPUT_SIZE*0.1), max_width=int(INPUT_SIZE*0.1),
                        min_holes=1, min_height=16, min_width=16, p=0.5),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ]),
    'val': A.Compose([
        A.Resize(int(INPUT_SIZE*1.15), int(INPUT_SIZE*1.15)),
        A.CenterCrop(INPUT_SIZE, INPUT_SIZE),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ]),
}

class AlbumentationsImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=None)
        self.transform = transform
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.transform: image = self.transform(image=image)['image']
        return image, target

image_datasets = {x: AlbumentationsImageFolder(os.path.join(working_data_dir, x), data_transforms[x]) for x in ['train','val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(f"Classes: {class_names}")
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
print(f"Training samples: {dataset_sizes['train']}, Validation samples: {dataset_sizes['val']}")

# Weighted sampler
train_targets = [s[1] for s in image_datasets['train'].samples]
class_sample_counts = np.bincount(train_targets, minlength=num_classes)
base_weights = 1.0 / torch.tensor(class_sample_counts, dtype=torch.float32)
samples_weight = torch.tensor([base_weights[t] for t in train_targets], dtype=torch.double)
sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
}

# ====================== FOCAL LOSS ======================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', weight=None):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
        self.weight = weight.to(DEVICE) if weight is not None else None
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean() if self.reduction=='mean' else focal_loss.sum()

criterion = FocalLoss(weight=base_weights)

# ====================== SWIN-T MODEL ======================
model = models.swin_t(weights='IMAGENET1K_V1')
num_ftrs = model.head.in_features
model.head = nn.Sequential(nn.Dropout(DROPOUT_RATE), nn.Linear(num_ftrs, num_classes))
model = model.to(DEVICE)

# ====================== OPTIMIZER & SCHEDULER ======================
pretrained_params = [p for n,p in model.named_parameters() if "head" not in n]
head_params = [p for n,p in model.named_parameters() if "head" in n]
optimizer = optim.AdamW([{'params': pretrained_params, 'lr':BASE_LR},
                         {'params': head_params, 'lr':HEAD_LR}],
                        weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# ====================== TTA ======================
def predict_with_tta(model, inputs, num_augments=TTA_AUGMENTS):
    model.eval(); preds=[]
    with torch.no_grad():
        preds.append(torch.softmax(model(inputs), dim=1))
        preds.append(torch.softmax(model(torch.flip(inputs,dims=[3])), dim=1))
        for scale in [0.9,1.1]:
            scaled = F.interpolate(inputs, scale_factor=scale, mode='bilinear', align_corners=False)
            if scale>1.0:
                start = (scaled.size(2)-inputs.size(2))//2
                scaled = scaled[:,:,start:start+inputs.size(2), start:start+inputs.size(3)]
            else:
                pad = (inputs.size(2)-scaled.size(2))//2
                scaled = F.pad(scaled,(pad,pad,pad,pad))
            preds.append(torch.softmax(model(scaled), dim=1))
    return torch.stack(preds[:num_augments]).mean(dim=0)

# ====================== TRAINING LOOP ======================
scaler = torch.amp.GradScaler(enabled=(DEVICE=="cuda"))
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0; patience_counter = 0
history = {'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[]}

print("="*60)
print("STARTING TRAINING")
print("="*60)

for epoch in range(NUM_EPOCHS):
    current_lr = scheduler.get_last_lr()[0]
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.2e}")

    # -------------------- TRAINING --------------------
    model.train()
    running_loss, running_corrects = 0.0, 0

    for batch_idx, (inputs, labels) in enumerate(dataloaders['train']):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        # Correct autocast context (works for both GPU and CPU)
        with torch.amp.autocast(device_type='cuda' if DEVICE.startswith('cuda') else 'cpu',
                                enabled=DEVICE.startswith('cuda')):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Skip invalid loss
        if not torch.isfinite(loss):
            print(f"⚠️ Skipping batch {batch_idx} due to invalid loss: {loss.item()}")
            continue

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels)

        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/{len(dataloaders['train'])} - Loss: {loss.item():.4f}")

    epoch_train_loss = running_loss / dataset_sizes['train']
    epoch_train_acc = (running_corrects.double() / dataset_sizes['train']).item()
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)

    # -------------------- VALIDATION --------------------
    model.eval()
    val_loss, val_corrects = 0.0, 0
    y_true, y_pred = [], []

    with torch.no_grad(), torch.amp.autocast(device_type='cuda' if DEVICE.startswith('cuda') else 'cpu',
                                            enabled=DEVICE.startswith('cuda')):
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            raw_outputs = model(inputs)
            loss = F.cross_entropy(raw_outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            # TTA (Test-Time Augmentation)
            final_outputs = predict_with_tta(model, inputs) if USE_TTA_IN_VALIDATION else raw_outputs
            _, preds = torch.max(final_outputs, 1)
            val_corrects += torch.sum(preds == labels)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    epoch_val_loss = val_loss / dataset_sizes['val']
    epoch_val_acc = (val_corrects.double() / dataset_sizes['val']).item()
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)

    print(f"train | Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f}")
    print(f"val   | Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f}")

    # -------------------- CHECKPOINTING --------------------
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "best_model_finegrained_swin.pth")
        print(f"✅ New best model saved (val acc: {best_acc:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"Early stopping patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("🛑 Early stopping triggered.")
            break

    # Scheduler update at end of epoch
    scheduler.step()

# ====================== PLOT LEARNING CURVE ======================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.title("Loss Curve"); plt.legend(); plt.grid(True)
plt.subplot(1,2,2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve"); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.savefig(GRAPH_SAVE_PATH); plt.show()
print(f"✅ Training curve saved to {GRAPH_SAVE_PATH}")

# ====================== CONFUSION MATRIX ======================
model.load_state_dict(torch.load("best_model_finegrained_swin.pth"))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for inputs, labels in dataloaders['val']:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = predict_with_tta(model, inputs) if USE_TTA_IN_VALIDATION else model(inputs)
        _, preds = torch.max(outputs,1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
plt.savefig("confusion_matrix_swin.png"); plt.show()
print("✅ Confusion matrix saved to confusion_matrix_swin.png")
