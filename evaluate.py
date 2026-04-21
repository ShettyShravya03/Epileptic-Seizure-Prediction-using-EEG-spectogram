"""
evaluate.py
-----------
Loads a trained EEGCNNBiLSTM checkpoint and runs Grad-CAM visualisation
across all 22 EEG channels for a single EDF segment.

Usage:
    python evaluate.py \
        --model_path  best_model_fold1.pth \
        --folder_path /path/to/segment_folder \
        --mean 0.613 0.628 0.606 \
        --std  0.317 0.314 0.316

The segment folder must contain exactly 22 spectrogram PNG images
named channel_<CHANNEL>.png (e.g. channel_FP1-F7.png).
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import EEGCNNBiLSTM


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def apply_gradcam(model, input_tensor, target_layer):
    """
    Compute Grad-CAM heatmap for a single-channel input through target_layer.

    Args:
        model:        EEGCNNBiLSTM instance (eval mode)
        input_tensor: [1, 22, 3, H, W]  — full 22-channel input
        target_layer: nn.Module to hook (typically last Conv layer in CNN encoder)

    Returns:
        list of np.ndarray, one normalised heatmap per channel  [H, W] in [0, 1]
    """
    activations, gradients = [], []

    def fwd_hook(_, __, output): activations.append(output)
    def bwd_hook(_, __, grad_out): gradients.append(grad_out[0])

    h_fwd = target_layer.register_forward_hook(fwd_hook)
    h_bwd = target_layer.register_backward_hook(bwd_hook)

    model.eval()
    output = model(input_tensor)
    model.zero_grad()
    output.backward()

    h_fwd.remove()
    h_bwd.remove()

    # Build per-channel heatmaps from pooled gradients
    heatmaps = []
    B, T, C, H, W = input_tensor.size()
    for i in range(T):
        grad = gradients[0][i]        # [C, H, W]
        act  = activations[0][i]      # [C, H, W]
        weights = grad.mean(dim=(1, 2))
        cam = torch.zeros(act.shape[1:]).to(input_tensor.device)
        for j, w in enumerate(weights):
            cam += w * act[j]
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        heatmaps.append(cam.detach().cpu().numpy())

    return heatmaps


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_segment(folder_path, mean, std, img_size=128):
    """Load all channel spectrograms from folder_path into a [1, 22, 3, H, W] tensor."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    if len(files) != 22:
        raise ValueError(f"Expected 22 spectrogram images, found {len(files)}")
    images = [transform(Image.open(os.path.join(folder_path, f)).convert('RGB')) for f in files]
    return torch.stack(images).unsqueeze(0), files  # [1, 22, 3, H, W]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  required=True)
    parser.add_argument('--folder_path', required=True)
    parser.add_argument('--mean', nargs=3, type=float, default=[0.613, 0.628, 0.606])
    parser.add_argument('--std',  nargs=3, type=float, default=[0.317, 0.314, 0.316])
    parser.add_argument('--threshold',   type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = EEGCNNBiLSTM().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Load segment
    input_tensor, filenames = load_segment(args.folder_path, args.mean, args.std)
    input_tensor = input_tensor.to(device)

    # Prediction
    with torch.no_grad():
        prob = torch.sigmoid(model(input_tensor)).item()

    severity = 'Mild' if prob < 0.5 else ('Moderate' if prob < 0.75 else 'Severe')
    label    = 'Preictal' if prob >= args.threshold else 'Interictal'
    print(f"Predicted probability : {prob:.4f}")
    print(f"Classification        : {label}")
    print(f"Severity              : {severity}")

    # Grad-CAM — target the last Conv layer in SimpleCNN encoder
    target_layer = model.cnn.encoder[-3]  # Conv2d(16, 32, 3)
    heatmaps = apply_gradcam(model, input_tensor, target_layer)

    # Plot 22-channel heatmap grid
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    axes = axes.flatten()

    for i, (heatmap, fname) in enumerate(zip(heatmaps, filenames)):
        orig = np.array(
            Image.open(os.path.join(args.folder_path, fname)).resize((128, 128))
        )
        heatmap_resized = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        colormap = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay  = cv2.addWeighted(orig, 0.6, colormap, 0.4, 0)

        axes[i].imshow(overlay)
        axes[i].axis('off')
        channel_name = fname.replace('channel_', '').replace('.png', '')
        axes[i].set_title(channel_name, fontsize=7)

    # Hide unused subplots
    for j in range(len(filenames), len(axes)):
        axes[j].axis('off')

    plt.suptitle(
        f"Grad-CAM — {label} (p={prob:.3f}, severity={severity})",
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig('gradcam_output.png', dpi=150)
    plt.show()
    print("✅ Grad-CAM saved → gradcam_output.png")


if __name__ == '__main__':
    main()