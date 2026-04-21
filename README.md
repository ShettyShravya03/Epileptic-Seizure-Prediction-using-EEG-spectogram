# Epileptic Seizure Prediction using EEG Spectrograms

> **IEEE COSMIC 2025** — Peer-reviewed and published

[![IEEE Paper](https://img.shields.io/badge/IEEE-COSMIC_2025-00629B?style=flat-square&logo=ieee&logoColor=white)](https://doi.org/10.1109/COSMIC67569.2025.11380836)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Results](#-key-results)
- [Model Architecture](#-model-architecture)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Interpretability](#-interpretability)
- [Publication](#-publication)
- [License](#-license)

---

## 🎯 Overview

A deep learning pipeline for **early epileptic seizure prediction** from multi-channel EEG signals, peer-reviewed and published at **IEEE COSMIC 2025**. The system converts raw 22-channel EEG into **Morlet wavelet spectrograms** (CWT) and processes them through a CNN-BiLSTM architecture with an attention mechanism to identify preictal (pre-seizure) states.

**Key innovation:** Morlet wavelet CWT captures non-stationary preictal EEG patterns with superior time-frequency resolution compared to STFT. The attention module learns to weight the most diagnostically relevant frequency bands per channel — improving both accuracy and clinical interpretability through Grad-CAM visualisations.

---

## 📊 Key Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **Wavelet-CNN-BiLSTM + Attention (ours)** | **85%** | **0.90** |
| CNN only (baseline) | 78% | 0.82 |
| LSTM only (baseline) | 80% | 0.84 |

- Dataset: **CHB-MIT Scalp EEG** (24 patients, 198 seizures, 664 hours)
- Validation: **10-fold subject-wise GroupKFold** (no data leakage across patients)
- Threshold optimisation: Youden's J statistic
- Severity classification: Mild · Moderate · Severe

---

## 🧠 Model Architecture

```
Raw EEG (22 channels, 256 Hz)
          │
          ▼
Morlet Wavelet CWT
(time-frequency spectrograms per channel)
Superior resolution over STFT for
non-stationary preictal patterns
          │
          ▼
3-Layer CNN
(spatial feature extraction from spectrograms)
          │
          ▼
2-Layer BiLSTM
(temporal sequence modelling across EEG channels)
          │
          ▼
Attention Mechanism
(focuses on seizure-relevant frequency bands)
          │
          ▼
Binary Classification
(Interictal vs Preictal)
          │
          ▼
Grad-CAM + SHAP
(per-channel activation maps for clinical interpretability)
```

---

## ✨ Features

- **Morlet wavelet CWT spectrograms** — superior time-frequency resolution over STFT for preictal EEG
- **Multi-channel EEG processing** — 22 standard bipolar EEG channels
- **CNN-BiLSTM + Attention** — captures spatial spectrogram patterns and temporal channel dependencies
- **Subject-wise GroupKFold** — prevents patient data leakage across train/test splits
- **Automatic threshold optimisation** — Youden's J statistic for optimal classification cutoff
- **Severity classification** — Mild, Moderate, and Severe seizure risk categories
- **Grad-CAM interpretability** — per-channel spectrogram activation maps for clinical transparency
- **Comprehensive metrics** — accuracy, ROC-AUC, confusion matrix, per-class precision/recall

---

## 🔧 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/ShettyShravya03/Epileptic-Seizure-Prediction-using-EEG-spectrogram.git
cd Epileptic-Seizure-Prediction-using-EEG-spectrogram

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt includes:** `torch` `torchvision` `scikit-learn` `pandas` `numpy` `matplotlib` `seaborn` `Pillow` `mne` `scipy`

```bash
# Optional: GPU acceleration (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Usage

### 1. Preprocess EEG → CWT spectrograms

```bash
python preprocess.py --data_dir ./chbmit --output_dir ./spectrograms
```

### 2. Train the model

```bash
python train.py --epochs 50 --batch_size 32 --folds 10
```

### 3. Evaluate and generate Grad-CAM maps

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --visualize
```

### 4. Predict on a new EEG sample

```python
from model import SeizurePredictor

predictor = SeizurePredictor(checkpoint="checkpoints/best_model.pt")
result = predictor.predict("path/to/eeg_sample.edf")

print(result["label"])       # "Preictal" or "Interictal"
print(result["severity"])    # "Mild", "Moderate", or "Severe"
print(result["confidence"])  # e.g. 0.91
```

---

## 📁 Dataset

**CHB-MIT Scalp EEG Database** — [PhysioNet](https://physionet.org/content/chbmit/1.0.0/)

| Property | Value |
|----------|-------|
| Patients | 24 paediatric patients |
| Seizures | 198 |
| Recording duration | 664 hours |
| Sampling rate | 256 Hz |
| Channels | 22 (bipolar pairs) |

> The dataset is publicly available on PhysioNet. Download and place it in `./chbmit` before running preprocessing.

---

## 🔍 Interpretability

Grad-CAM activation maps are generated per EEG channel to highlight the spectrogram time-frequency regions most influential to the seizure prediction. This transparency is essential for clinical deployment — a clinician can verify which frequency bands and time windows the model flagged as preictal.

Example output per channel:
- Heatmap overlay on the CWT spectrogram
- Highlighted frequency bands (delta, theta, alpha, beta, gamma)
- Per-channel contribution scores via SHAP

---

## 📄 Publication

This work was peer-reviewed and published at **IEEE COSMIC 2025**.

```bibtex
@inproceedings{shetty2025seizure,
  title     = {Wavelet-Based Deep Neural Network with Attention for Interpretable
               Epileptic Seizure Prediction Using EEG Spectrograms},
  author    = {Shetty, Shravya S and {others}},
  booktitle = {Proceedings of IEEE COSMIC 2025},
  year      = {2025},
  doi       = {10.1109/COSMIC67569.2025.11380836}
}
```

---

## 📜 License

MIT © 2025 Shravya S Shetty
