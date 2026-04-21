import os
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import torchvision.transforms.functional as F
import random
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GroupKFold
from torch import amp

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset Class
class EEGSpectrogramDataset(Dataset):
    def __init__(self, dataframe, transform = None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.channels = [col for col in self.df.columns if col.startswith('channel_')]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = []
        for ch in self.channels:
            img_path = row[ch]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)  # Each img: [3, H, W]
        label = torch.tensor(int(row['label']), dtype=torch.float32)
        return torch.stack(images), label  # Shape: [22, 3, H, W]


# Image transforms
transform_no_norm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Compute mean and std
def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    for images, _ in loader:
        images = images.view(images.size(0), 3, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += images.size(0)

    mean /= total_images
    std /= total_images
    return mean.tolist(), std.tolist()

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        attn_weights = torch.softmax(self.attn(lstm_output).squeeze(-1), dim=1)
        context = torch.sum(lstm_output * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights

class SimpleCNN(nn.Module):
    def __init__(self, out_dim = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,16,3), nn.ReLU(), nn.MaxPool2d(2),  # 111x111
            nn.Conv2d(16,32,3), nn.ReLU(), nn.MaxPool2d(2), # 54x54
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, out_dim)  # Projection layer
        self.out_dim = out_dim
    
    def forward(self, x):
        x = self.encoder(x).squeeze(-1).squeeze(-1)  # [B, 32]
        x = self.fc(x)  # [B, out_dim]
        return x


class EEGCNNBiLSTM(nn.Module):
    def __init__(self, cnn_out_dim=128, hidden_dim=64):
        super(EEGCNNBiLSTM, self).__init__()
        self.cnn = SimpleCNN(out_dim=cnn_out_dim)
        self.lstm = nn.LSTM(input_size=cnn_out_dim, hidden_size=hidden_dim, 
                            num_layers=2, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):  # x: [B, 22, 3, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)           # [B*T, 3, H, W]
        feats = self.cnn(x)                  # [B*T, 128]
        feats = feats.view(B, T, -1)         # [B, 22, 128]

        lstm_out, _ = self.lstm(feats)       # [B, 22, 2*hidden_dim]
        context, attn_weights = self.attn(lstm_out)  # [B, 2*hidden_dim]
        logits = self.fc(context).squeeze(1) # [B]
        return logits

def main(): 
    # Load and preprocess CSV
    df = pd.read_csv("C:/Users/ranji/OneDrive/Desktop/preprocessed_dataset.csv")  # Replace with actual path
    df.replace(r'\\', "/", regex=True, inplace=True)   
    
    best_model_wts = None
    best_accuracy = 0.0
    best_fold = -1

    # Parameters
    n_splits = 10  # Number of folds for GroupKFold
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-4  # Safe baseline for all layers
    weight_decay = 1e-4


    # Store overall metrics
    fold_accuracies = []
    fold_classification_reports = []
    fold_confusion_matrices = []

    # Prepare group and labels for GroupKFold
    groups = df["patient_id"].values  # Group column (e.g., patients)
    labels = df["label"].values     # Your label column

    gkf = GroupKFold(n_splits=n_splits)

    # Image transforms with normalization will be defined after computing mean and std

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, y=labels, groups=groups)):
        print(f"\n{'='*20} Fold {fold+1} / {n_splits} {'='*20}")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Compute mean and std for normalization
        temp_dataset = EEGSpectrogramDataset(dataframe=train_df, transform=transform_no_norm)
        computed_mean, computed_std = compute_mean_std(temp_dataset)
        print(f"📊 Computed Mean: {computed_mean}")
        print(f"📊 Computed Std: {computed_std}")

        transform_train = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=computed_mean, std=computed_std),
        ])

        # validation transform without data augmentation
        transform_val = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=computed_mean, std=computed_std),
        ])



        # Create Datasets and DataLoaders
        train_dataset = EEGSpectrogramDataset(dataframe=train_df, transform=transform_train)
        val_dataset = EEGSpectrogramDataset(dataframe=val_df, transform=transform_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

        # Model, Loss, Optimizer
        model = EEGCNNBiLSTM().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

        scaler = amp.GradScaler()

        # Initialize lists to store loss and learning rates
        epoch_losses = []
        learning_rates = []

        # Training Loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, labels_batch in train_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                optimizer.zero_grad()

                # AMP
                with amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels_batch)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            epoch_losses.append(avg_train_loss)

            # Calculate validation loss
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for images, labels_batch in val_loader:
                    images, labels_batch = images.to(device), labels_batch.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels_batch)
                    val_running_loss += loss.item()

            avg_val_loss = val_running_loss / len(val_loader)

            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            # Step scheduler with validation loss - epoch level
            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Learning Rate: {current_lr}")


        # Evaluation
        model.eval()
        preds = []
        true_labels = []
        severity_levels = []
        all_probs = []  # Store all probabilities for ROC curve

        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                # labels_batch = labels_batch.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy())  # <-- Add this line
                # predicted = (probs.cpu().numpy() > 0.5).astype(int)
                # preds.append(predicted)
                true_labels.extend(labels_batch.numpy())

                # Severity Analysis
                for prob in probs.cpu().numpy():
                    if prob < 0.5:
                        severity = "Mild"
                    elif prob < 0.75:
                        severity = "Moderate"
                    else:
                        severity = "Severe"
                    severity_levels.append(severity)

        # preds = np.concatenate(preds).flatten()
        true_labels = np.array(true_labels).flatten()

        # Display Severity Levels
        for i, severity in enumerate(severity_levels):
            print(f"Sample {i+1}: Severity Level = {severity}")

        # Plot Loss vs. Epoch
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label="Loss")
        plt.title(f"Loss vs. Epoch - Fold {fold+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()
        plt.show()

        # Plot Learning Rate vs. Epoch
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(learning_rates) + 1), learning_rates, marker='o', color='orange', label="Learning Rate")
        plt.title(f"Learning Rate vs. Epoch - Fold {fold+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.grid()
        plt.legend()
        plt.show()

        fpr, tpr, thresholds = roc_curve(true_labels, all_probs)
        auc_score = roc_auc_score(true_labels, all_probs)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid()
        plt.show()

        # Find the best threshold using Youden's J statistic
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]
        print(f"✅ Best Threshold: {best_threshold:.3f}")

        final_preds = (np.array(all_probs) > best_threshold).astype(int)
        acc = accuracy_score(true_labels, final_preds)
        print(f"\n🧠 Fold {fold+1} Accuracy: {acc*100:.2f}%")



        print(classification_report(true_labels, final_preds))

        cm = confusion_matrix(true_labels, final_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - Fold {fold+1}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()



        if acc > best_accuracy:
            best_accuracy = acc
            best_model_wts = deepcopy(model.state_dict())
            best_fold = fold + 1  # human-readable

    if best_model_wts is not None:
        model_path = f"resnet_bilstm_attention_best_fold{best_fold}.pth"
        torch.save(best_model_wts, model_path)
        print(f"\n✅ Best model from Fold {best_fold} with Accuracy: {best_accuracy*100:.2f}% saved to: {model_path}")

if __name__ == '__main__':
    main()

