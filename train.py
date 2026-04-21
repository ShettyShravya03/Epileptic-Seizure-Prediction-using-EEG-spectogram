"""
train.py
--------
Train EEGCNNBiLSTM with 10-fold subject-wise GroupKFold cross-validation
on the preprocessed CHB-MIT spectrogram dataset.

Subject-wise splitting (GroupKFold on patient_id) prevents EEG data from the
same patient appearing in both train and validation, avoiding data leakage.

Usage:
    python train.py --csv preprocessed_dataset.csv --epochs 10 --folds 10

Outputs:
    best_model_fold<N>.pth   — weights of the fold with highest accuracy
"""

import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy
from PIL import Image
from torchvision import transforms
from torch import nn, optim, amp
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

from model import EEGCNNBiLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EEGSpectrogramDataset(Dataset):
    """Loads 22-channel CWT spectrogram images for one EDF segment."""

    def __init__(self, dataframe, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform
        self.channels  = [c for c in self.df.columns if c.startswith('channel_')]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        images = []
        for ch in self.channels:
            img = Image.open(row[ch]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        label = torch.tensor(int(row['label']), dtype=torch.float32)
        return torch.stack(images), label  # [22, 3, H, W], scalar


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def compute_mean_std(dataset):
    """Compute per-channel mean and std over the training split."""
    loader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=0)
    mean   = torch.zeros(3)
    std    = torch.zeros(3)
    total  = 0
    for images, _ in loader:
        images = images.view(images.size(0), 3, -1)
        mean  += images.mean(2).sum(0)
        std   += images.std(2).sum(0)
        total += images.size(0)
    return (mean / total).tolist(), (std / total).tolist()


_base_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def make_transforms(mean, std):
    t = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return t, t  # train and val use same transforms (no augmentation here)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_fold(train_df, val_df, args):
    # Compute normalisation stats from training split only
    tmp_ds             = EEGSpectrogramDataset(train_df, transform=_base_transform)
    mean, std          = compute_mean_std(tmp_ds)
    transform, _       = make_transforms(mean, std)

    train_ds = EEGSpectrogramDataset(train_df, transform=transform)
    val_ds   = EEGSpectrogramDataset(val_df,   transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    model     = EEGCNNBiLSTM().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    scaler    = amp.GradScaler()

    epoch_losses, val_losses, lrs = [], [], []

    for epoch in range(args.epochs):
        # --- train ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with amp.autocast(device_type='cuda'):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_train = running_loss / len(train_loader)
        epoch_losses.append(avg_train)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_loss += criterion(model(images), labels).item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        print(f"  Epoch {epoch+1:>2}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}  lr={current_lr}")

    return model, epoch_losses, val_losses, lrs


def evaluate_fold(model, val_df, transform, fold, args):
    val_ds     = EEGSpectrogramDataset(val_df, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_probs, true_labels, severity_levels = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            all_probs.extend(probs)
            true_labels.extend(labels.numpy())
            for p in probs:
                severity_levels.append('Mild' if p < 0.5 else ('Moderate' if p < 0.75 else 'Severe'))

    all_probs   = np.array(all_probs)
    true_labels = np.array(true_labels)

    # Optimal threshold via Youden's J
    fpr, tpr, thresholds = roc_curve(true_labels, all_probs)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    final_preds    = (all_probs > best_threshold).astype(int)
    acc            = accuracy_score(true_labels, final_preds)
    auc            = roc_auc_score(true_labels, all_probs)

    print(f"\n  Fold {fold} — Accuracy: {acc*100:.2f}%  AUC: {auc:.3f}  Threshold: {best_threshold:.3f}")
    print(classification_report(true_labels, final_preds))

    # Confusion matrix
    cm = confusion_matrix(true_labels, final_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Interictal', 'Preictal'],
                yticklabels=['Interictal', 'Preictal'])
    plt.title(f'Confusion Matrix — Fold {fold}')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(f'confusion_matrix_fold{fold}.png'); plt.close()

    return acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',        default='preprocessed_dataset.csv')
    parser.add_argument('--epochs',     type=int, default=10)
    parser.add_argument('--folds',      type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr',         type=float, default=1e-4)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.replace(r'\\', '/', regex=True, inplace=True)

    groups = df['patient_id'].values
    labels = df['label'].values
    gkf    = GroupKFold(n_splits=args.folds)

    best_acc, best_weights, best_fold = 0.0, None, -1

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, y=labels, groups=groups), start=1):
        print(f"\n{'='*40}\nFold {fold} / {args.folds}\n{'='*40}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        model, train_losses, val_losses, lrs = train_one_fold(train_df, val_df, args)

        # Recompute transform with fold's stats for evaluation
        tmp_ds    = EEGSpectrogramDataset(train_df, transform=_base_transform)
        mean, std = compute_mean_std(tmp_ds)
        transform, _ = make_transforms(mean, std)

        acc = evaluate_fold(model, val_df, transform, fold, args)

        if acc > best_acc:
            best_acc     = acc
            best_weights = deepcopy(model.state_dict())
            best_fold    = fold

    # Save best model
    save_path = f'best_model_fold{best_fold}.pth'
    torch.save(best_weights, save_path)
    print(f"\n✅ Best model: Fold {best_fold} — Accuracy {best_acc*100:.2f}%  →  {save_path}")


if __name__ == '__main__':
    main()