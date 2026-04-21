<!-- BANNER — upload banner.svg to this repo root -->
<p align="center">
  <img src="./banner.svg" width="100%" alt="Epileptic Seizure Prediction using EEG Spectrograms">
</p>

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11380836"><img src="https://img.shields.io/badge/IEEE-COSMIC_2025-00629B?style=for-the-badge&logo=ieee&logoColor=white"></a>&nbsp;
  <img src="https://img.shields.io/badge/Accuracy-85%25-7c3aed?style=for-the-badge">&nbsp;
  <img src="https://img.shields.io/badge/ROC--AUC-0.90-a855f7?style=for-the-badge">&nbsp;
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>&nbsp;
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>&nbsp;
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge">
</p>

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

---

## 🎯 Overview

A deep learning pipeline for **early epileptic seizure prediction** from multi-channel EEG signals, peer-reviewed and published at **IEEE COSMIC 2025**. The system converts raw 22-channel EEG into **Morlet wavelet spectrograms** (CWT) and processes them through a CNN-BiLSTM architecture with an attention mechanism to identify preictal (pre-seizure) states.

**Key innovation:** Morlet wavelet CWT captures non-stationary preictal EEG patterns with superior time-frequency resolution compared to STFT. The attention module learns to weight the most diagnostically relevant frequency bands per channel — improving both accuracy and clinical interpretability through Grad-CAM visualisations.

---

## 📊 Key Results

<table>
  <tr>
    <th>Model</th>
    <th>Accuracy</th>
    <th>ROC-AUC</th>
    <th>Improvement</th>
  </tr>
  <tr>
    <td><strong>Wavelet-CNN-BiLSTM + Attention (ours)</strong></td>
    <td><strong>85%</strong></td>
    <td><strong>0.90</strong></td>
    <td>—</td>
  </tr>
  <tr>
    <td>CNN only (baseline)</td>
    <td>78%</td>
    <td>0.82</td>
    <td>+7% acc</td>
  </tr>
  <tr>
    <td>LSTM only (baseline)</td>
    <td>80%</td>
    <td>0.84</td>
    <td>+5% acc</td>
  </tr>
</table>

- **Dataset:** CHB-MIT Scalp EEG (24 patients, 198 seizures, 664 hours)
- **Validation:** 10-fold subject-wise GroupKFold — no data leakage across patients
- **Threshold optimisation:** Youden's J statistic
- **Severity classification:** Mild · Moderate · Severe

---

## 🧠 Model Architecture

```
Raw EEG (22 channels, 256 Hz)
          │
          ▼
  ┌───────────────────┐
  │  Morlet Wavelet   │  CWT spectrograms — superior time-frequency
  │  CWT Transform    │  resolution for non-stationary preictal patterns
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │   3-Layer CNN     │  Spatial feature extraction from spectrograms
  │  (Conv + BN + ReLU│  per EEG channel
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │  2-Layer BiLSTM   │  Temporal modelling — forward + backward
  │                   │  across EEG channel sequences
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │ Attention Module  │  Learns to focus on seizure-relevant
  │                   │  frequency bands and time windows
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │ Binary Classifier │  Interictal vs Preictal
  │  + Severity       │  Mild / Moderate / Severe
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │ Grad-CAM + SHAP   │  Per-channel activation maps
  │ Interpretability  │  for clinical transparency
  └───────────────────┘
```

---

## ✨ Features

- **Morlet wavelet CWT spectrograms** — superior time-frequency resolution over STFT for preictal EEG
- **22-channel multi-channel EEG processing** — standard bipolar montage
- **CNN-BiLSTM + Attention** — captures spatial spectrogram patterns + temporal dependencies
- **Subject-wise GroupKFold** — prevents patient data leakage across train/test splits
- **Automatic threshold optimisation** — Youden's J statistic for optimal classification cutoff
- **Severity classification** — Mild, Moderate, and Severe seizure risk categories
- **Grad-CAM interpretability** — per-channel spectrogram activation maps for clinical transparency
- **Comprehensive metrics** — accuracy, ROC-AUC, confusion matrix, per-class precision/recall

---

## 🔧 Installation

**Requirements:** Python 3.8+ · PyTorch 2.0+ · CUDA-compatible GPU (recommended)

```bash
# Clone the repository
git clone https://github.com/ShettyShravya03/Epileptic-Seizure-Prediction-using-EEG-spectrogram.git
cd Epileptic-Seizure-Prediction-using-EEG-spectrogram

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` includes: `torch` `torchvision` `scikit-learn` `pandas` `numpy` `matplotlib` `seaborn` `Pillow` `mne` `scipy`

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

> Download the dataset from PhysioNet and place it in `./chbmit` before running preprocessing.

---

## 🔍 Interpretability

Grad-CAM activation maps are generated per EEG channel to highlight the spectrogram time-frequency regions most influential to the prediction. This is essential for clinical deployment — a clinician can verify exactly which frequency bands and time windows the model flagged as preictal.

**Output per channel:**
- Heatmap overlay on the CWT spectrogram
- Highlighted frequency bands (delta, theta, alpha, beta, gamma)
- Per-channel contribution scores via SHAP

---

## 📄 Publication

This work was peer-reviewed and published at the **2025 Second International Conference on Computing, Communication, and Smart Systems (IEEE COSMIC 2025)**.

**Authors:** Abhishek S. Rao · Ramaprasad Poojary · Shravya A. Prabhu · **Shravya S. Shetty** · Diya M. Shetty · H. Nagesh Shenoy

**DOI:** [10.1109/COSMIC67569.2025.11380836](https://ieeexplore.ieee.org/document/11380836)

```bibtex
@inproceedings{rao2025seizure,
  title     = {Wavelet-Based Deep Neural Network with Attention for Interpretable
               Epileptic Seizure Prediction Using EEG Spectrograms},
  author    = {Rao, Abhishek S. and Poojary, Ramaprasad and Prabhu, Shravya A.
               and Shetty, Shravya S. and Shetty, Diya M. and Shenoy, H. Nagesh},
  booktitle = {2025 Second International Conference on Computing, Communication,
               and Smart Systems (COSMIC)},
  year      = {2025},
  doi       = {10.1109/COSMIC67569.2025.11380836}
}
```

---

## 🏷️ Topics

`eeg` `seizure-prediction` `deep-learning` `bilstm` `wavelet-transform` `cnn` `attention-mechanism` `grad-cam` `medical-ai` `chb-mit` `signal-processing` `ieee` `pytorch` `explainable-ai`

---

<p align="center">
  <sub>Built by <a href="https://github.com/ShettyShravya03">Shravya S Shetty</a> et al. · NMAM Institute of Technology · IEEE COSMIC 2025</sub>
</p>
