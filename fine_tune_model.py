import os
import time
import copy
import json
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import platform

def main():
    # ====================== CONFIGURATION ======================
    DATA_DIR = "ppe_dataset"
    MODEL_SAVE_PATH = "best_ppe_classifier.pth"
    GRAPH_SAVE_PATH = "learning_curve.png"
    NUM_EPOCHS = 200
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EARLY_STOPPING_PATIENCE = 10
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTHING = 0.1
    DROPOUT_RATE = 0.4
    # Automatically set NUM_WORKERS to 0 on Windows to avoid multiprocessing spawn errors
    NUM_WORKERS = 0 if platform.system() == "Windows" else 4
    LOG_CSV = "training_log.csv"
    SAVE_CLASS_MAP = "class_to_idx.json"
    GRAD_CLIP_NORM = 1.0
    RANDOM_SEED = 42

    print(f"--- Fine-Tuning EfficientNet-B0 (Small Dataset Optimized) ---")
    print(f"Using device: {DEVICE}")

    # ====================== REPRODUCIBILITY ======================
    def set_seed(seed=RANDOM_SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    set_seed()

    # ====================== DATA AUGMENTATION ======================
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # ====================== DATA LOADING (with imbalance handling) ======================
    # Basic checks for dataset structure
    for split in ['train', 'val']:
        split_path = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Expected dataset directory missing: {split_path}")

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                    for x in ['train', 'val']}

    # Save class_to_idx for inference usage
    with open(SAVE_CLASS_MAP, "w") as f:
        json.dump(image_datasets['train'].class_to_idx, f)
    print(f"Saved class map to: {SAVE_CLASS_MAP}")

    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # detect imbalance and build sampler if needed
    train_targets = [s[1] for s in image_datasets['train'].samples]
    class_sample_counts = np.bincount(train_targets, minlength=num_classes)
    print("Train class distribution:", dict(zip(class_names, class_sample_counts)))

    use_sampler = False
    sampler = None
    # If max/min ratio > 1.5 and more than 1 class, enable weighted sampler
    if num_classes > 1:
        max_count = class_sample_counts.max()
        min_count = class_sample_counts.min()
        if min_count > 0 and (max_count / (min_count + 1e-6)) > 1.5:
            weights = 1.0 / class_sample_counts
            samples_weight = np.array([weights[t] for t in train_targets])
            samples_weight = torch.from_numpy(samples_weight).double()
            sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
            use_sampler = True
            print("Using WeightedRandomSampler to mitigate class imbalance.")
        else:
            print("Class distribution is relatively balanced; not using sampler.")

    # Build dataloaders
    if use_sampler:
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS),
            'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        }
    else:
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
            'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"Training samples: {dataset_sizes['train']}, Validation samples: {dataset_sizes['val']}")

    # ====================== MODEL SETUP ======================
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    # Gradual unfreezing — initially only top blocks trainable
    for name, param in model.features.named_parameters():
        param.requires_grad = '6' in name or '7' in name  # unfreeze last 2 blocks

    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(DEVICE)

    # ====================== LOSS, OPTIMIZER, SCHEDULER ======================
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # ====================== TRAINING LOOP (with AMP, grad clip, progressive unfreeze) ======================
    scaler = torch.amp.GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # prepare CSV logging
    with open(LOG_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'])

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print('-' * 40)

        # progressive unfreezing example:
        if epoch == 10:
            for name, param in model.features.named_parameters():
                if 'blocks.5' in name:
                    param.requires_grad = True
            print("🔓 Unfroze EfficientNet block 5 for deeper fine-tuning.")
        if epoch == 20:
            for name, param in model.features.named_parameters():
                if 'blocks.4' in name:
                    param.requires_grad = True
            print("🔓 Unfroze EfficientNet block 4 for deeper fine-tuning.")

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            loader = dataloaders[phase]
            for inputs, labels in loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        # gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / max(1, dataset_sizes[phase])
            epoch_acc = running_corrects.double() / max(1, dataset_sizes[phase])

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f"{phase} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # scheduler step once per epoch (after training phase)
            if phase == 'train':
                scheduler.step()

            # validation checks
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"✅ New best model saved (val acc: {best_acc:.4f})")

                # Early stopping by val loss
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"Early Stopping patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        # CSV logging
        with open(LOG_CSV, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                epoch + 1,
                history['train_loss'][-1] if history['train_loss'] else '',
                history['train_acc'][-1] if history['train_acc'] else '',
                history['val_loss'][-1] if history['val_loss'] else '',
                history['val_acc'][-1] if history['val_acc'] else ''
            ])

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("\n🛑 Early stopping triggered.")
            break

    time_elapsed = time.time() if 'since' in globals() else 0
    print(f"\nTraining finished. Best val acc: {best_acc:.4f}")

    # load best weights
    model.load_state_dict(best_model_wts)

    # ====================== PERFORMANCE VISUALIZATION ======================
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy')
    plt.legend(); plt.title('Accuracy Over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.title('Loss Over Epochs')

    plt.tight_layout()
    plt.savefig(GRAPH_SAVE_PATH)
    print(f"📊 Training curve saved to {GRAPH_SAVE_PATH}")

    # ====================== CONFUSION MATRIX EVALUATION ======================
    print("\n--- Validation Set Evaluation ---")
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix (Validation)')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()