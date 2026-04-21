"""
experiments/resnet_attention.py
--------------------------------
Experimental variant: ResNet18 + Squeeze-and-Excitation attention
with StratifiedKFold (5 folds) and SpecAugment data augmentation.

This was explored as an alternative to the CNN-BiLSTM approach.
The final submitted model uses EEGCNNBiLSTM (see model.py / train.py).

To run:
    python experiments/resnet_attention.py --csv preprocessed_dataset.csv
"""

import argparse
import random
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy
from PIL import Image
from torchvision import transforms, models
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_curve, roc_auc_score)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# SpecAugment — time and frequency masking for EEG spectrograms
# ---------------------------------------------------------------------------

class SpecAugment:
    """Apply random time and frequency masks to a spectrogram tensor."""

    def __init__(self, time_mask_param=20, freq_mask_param=10,
                 num_time_masks=1, num_freq_masks=1):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks  = num_time_masks
        self.num_freq_masks  = num_freq_masks

    def __call__(self, img):
        img = img.clone()
        for _ in range(self.num_freq_masks):
            band  = random.randint(0, self.freq_mask_param)
            start = random.randint(0, img.shape[1] - band)
            img[:, start:start + band, :] = 0
        for _ in range(self.num_time_masks):
            band  = random.randint(0, self.time_mask_param)
            start = random.randint(0, img.shape[2] - band)
            img[:, :, start:start + band] = 0
        return img


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EEGSpectrogramDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform
        self.channels  = [c for c in self.df.columns if c.startswith('channel_')]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = []
        for ch in self.channels:
            img = Image.open(row[ch]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        image_tensor = torch.cat(images, dim=0)  # [66, H, W] (22 * 3)
        label = torch.tensor(int(row['label']), dtype=torch.float32)
        return image_tensor, label


# ---------------------------------------------------------------------------
# ResNet18 + Squeeze-and-Excitation Attention
# ---------------------------------------------------------------------------

class ResNetAttention(nn.Module):
    """
    ResNet18 with a modified first Conv layer (66 input channels for 22 * RGB)
    and a Squeeze-and-Excitation attention block before the final classifier.
    """

    def __init__(self, input_channels=66, dropout_prob=0.5):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        self.resnet.fc = nn.Identity()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 512 // 16, 1), nn.ReLU(),
            nn.Conv2d(512 // 16, 512, 1), nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc      = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)       # [B, 512, H, W]
        x = x * self.se(x)              # SE channel attention
        x = self.resnet.avgpool(x)
        x = self.dropout(torch.flatten(x, 1))
        return self.fc(x).squeeze(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_base_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=0)
    mean, std, total = torch.zeros(3), torch.zeros(3), 0
    for images, _ in loader:
        images = images.view(images.size(0), 3, -1)
        mean  += images.mean(2).sum(0)
        std   += images.std(2).sum(0)
        total += images.size(0)
    return (mean / total).tolist(), (std / total).tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',        default='preprocessed_dataset.csv')
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--folds',      type=int,   default=5)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--lr',         type=float, default=1e-4)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.replace(r'\\', '/', regex=True, inplace=True)

    labels = df['label'].values
    skf    = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    best_acc, best_weights, best_fold = 0.0, None, -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels), start=1):
        print(f"\n{'='*40}\nFold {fold} / {args.folds}\n{'='*40}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df   = df.iloc[val_idx].reset_index(drop=True)

        tmp_ds    = EEGSpectrogramDataset(train_df, transform=_base_transform)
        mean, std = compute_mean_std(tmp_ds)

        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            SpecAugment(time_mask_param=20, freq_mask_param=10, num_time_masks=2, num_freq_masks=2),
            transforms.Normalize(mean=mean, std=std),
        ])
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        train_ds = EEGSpectrogramDataset(train_df, transform=transform_train)
        val_ds   = EEGSpectrogramDataset(val_df,   transform=transform_val)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

        model     = ResNetAttention().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr,
            steps_per_epoch=len(train_loader), epochs=args.epochs
        )

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for images, labels_batch in train_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), labels_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
            print(f"  Epoch {epoch+1}/{args.epochs}  loss={running_loss/len(train_loader):.4f}")

        # Evaluate
        model.eval()
        all_probs, true_labels = [], []
        with torch.no_grad():
            for images, labels_batch in val_loader:
                probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
                all_probs.extend(probs)
                true_labels.extend(labels_batch.numpy())

        all_probs   = np.array(all_probs)
        true_labels = np.array(true_labels)
        fpr, tpr, thresholds = roc_curve(true_labels, all_probs)
        best_threshold = thresholds[np.argmax(tpr - fpr)]
        preds = (all_probs > best_threshold).astype(int)
        acc   = accuracy_score(true_labels, preds)
        print(f"\n  Fold {fold} — Accuracy: {acc*100:.2f}%  AUC: {roc_auc_score(true_labels, all_probs):.3f}")

        if acc > best_acc:
            best_acc, best_weights, best_fold = acc, deepcopy(model.state_dict()), fold

    save_path = f'resnet_best_fold{best_fold}.pth'
    torch.save(best_weights, save_path)
    print(f"\n✅ Best ResNet model: Fold {best_fold}  Accuracy {best_acc*100:.2f}%  →  {save_path}")


if __name__ == '__main__':
    main()