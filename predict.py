"""
predict.py
----------
Run inference on a single EDF spectrogram segment using a trained
EEGCNNBiLSTM checkpoint.

Usage:
    python predict.py \
        --model_path  best_model_fold1.pth \
        --folder_path /path/to/segment_folder \
        --mean 0.613 0.628 0.606 \
        --std  0.317 0.314 0.316 \
        --threshold   0.501
"""

import os
import argparse
import torch
from PIL import Image
from torchvision import transforms

from model import EEGCNNBiLSTM


def load_segment(folder_path, mean, std, img_size=128):
    """
    Load 22 channel spectrogram PNGs from folder_path.

    Returns:
        torch.Tensor: [1, 22, 3, img_size, img_size]
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    if len(files) != 22:
        raise ValueError(f"Expected 22 spectrogram images, found {len(files)}")
    images = [
        transform(Image.open(os.path.join(folder_path, f)).convert('RGB'))
        for f in files
    ]
    return torch.stack(images).unsqueeze(0)  # [1, 22, 3, H, W]


def predict(model_path, folder_path, mean, std, threshold=0.5, device=None):
    """
    Predict seizure probability for a single EDF segment.

    Returns:
        dict with keys: probability, label, severity
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EEGCNNBiLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    input_tensor = load_segment(folder_path, mean, std).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(input_tensor)).item()

    label    = 'Preictal'   if prob >= threshold else 'Interictal'
    severity = 'Mild'       if prob < 0.5        else ('Moderate' if prob < 0.75 else 'Severe')

    return {'probability': prob, 'label': label, 'severity': severity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  required=True,  help='Path to .pth checkpoint')
    parser.add_argument('--folder_path', required=True,  help='Folder with 22 channel PNGs')
    parser.add_argument('--mean', nargs=3, type=float, default=[0.613, 0.628, 0.606])
    parser.add_argument('--std',  nargs=3, type=float, default=[0.317, 0.314, 0.316])
    parser.add_argument('--threshold',   type=float, default=0.501)
    args = parser.parse_args()

    result = predict(
        model_path=args.model_path,
        folder_path=args.folder_path,
        mean=args.mean,
        std=args.std,
        threshold=args.threshold,
    )

    print(f"Predicted probability : {result['probability']:.4f}")
    print(f"Classification        : {result['label']}")
    print(f"Severity              : {result['severity']}")


if __name__ == '__main__':
    main()