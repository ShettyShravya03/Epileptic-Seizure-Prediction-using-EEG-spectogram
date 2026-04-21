"""
model.py
--------
CNN-BiLSTM-Attention architecture for epileptic seizure prediction
from 22-channel EEG Morlet wavelet spectrograms.

Architecture:
    Input: [B, 22, 3, H, W]  (batch of 22-channel spectrogram stacks)
    → Per-channel SimpleCNN  → [B, 22, 128]  (spatial feature extraction)
    → 2-layer BiLSTM         → [B, 22, 128]  (temporal sequence modelling)
    → Attention              → [B, 128]       (focus on preictal patterns)
    → Linear                 → [B, 1]         (seizure probability)
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Additive attention over the LSTM time dimension."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: [B, T, hidden_dim]
        Returns:
            context:      [B, hidden_dim]
            attn_weights: [B, T]
        """
        attn_weights = torch.softmax(self.attn(lstm_output).squeeze(-1), dim=1)
        context = torch.sum(lstm_output * attn_weights.unsqueeze(-1), dim=1)
        return context, attn_weights


class SimpleCNN(nn.Module):
    """
    Lightweight per-channel CNN encoder.
    Encodes a single spectrogram [3, H, W] → feature vector [out_dim].
    """

    def __init__(self, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.encoder(x).squeeze(-1).squeeze(-1)  # [B, 32]
        return self.fc(x)                             # [B, out_dim]


class EEGCNNBiLSTM(nn.Module):
    """
    Full model: SimpleCNN per channel → BiLSTM across channels → Attention → classifier.

    Args:
        cnn_out_dim (int): Output dimension of the per-channel CNN.  Default 128.
        hidden_dim  (int): Hidden size of the BiLSTM (each direction).  Default 64.
    """

    def __init__(self, cnn_out_dim=128, hidden_dim=64):
        super().__init__()
        self.cnn  = SimpleCNN(out_dim=cnn_out_dim)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = Attention(hidden_dim * 2)
        self.fc   = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        """
        Args:
            x: [B, 22, 3, H, W]
        Returns:
            logits: [B]  (raw scores, apply sigmoid for probabilities)
        """
        B, T, C, H, W = x.size()
        feats = self.cnn(x.view(B * T, C, H, W)).view(B, T, -1)  # [B, 22, cnn_out_dim]
        lstm_out, _ = self.lstm(feats)                             # [B, 22, 2*hidden_dim]
        context, _  = self.attn(lstm_out)                         # [B, 2*hidden_dim]
        return self.fc(context).squeeze(1)                         # [B]