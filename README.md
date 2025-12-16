# Deepfake Detection

This repository contains the code and experiments for a comparative study of **deepfake detection methods** based on four major paradigms: **spatial**, **frequency-based**, **temporal**, and **transformer-based** approaches.  
The goal of this project is to evaluate the strengths and limitations of these paradigms under a **controlled and unified experimental setup**.

This work is developed as part of an academic project and emphasizes **reproducibility, fairness of comparison, and clarity of methodology**.

---

## Overview

Deepfake generation methods have become increasingly realistic, making reliable detection challenging. This project evaluates four representative deepfake detection paradigms:

1. Spatial-based detection using convolutional neural networks (CNNs)
2. Frequency-based detection using Fourier-derived features
3. Temporal-based detection using sequence modeling
4. Transformer-based detection using self-attention mechanisms

Each approach is trained and evaluated using appropriate datasets and a shared preprocessing pipeline to ensure comparability.

---

## Detection Paradigms

### 1. Spatial CNN (EfficientNet-B4)
- Operates on individual RGB face crops
- Learns texture-level and blending artifacts
- Fine-tuned EfficientNet-B4 architecture

### 2. Frequency-Based CNN (FreqNet)
- Extends RGB input with **Fourier-derived magnitude features**
- Uses high-pass and multi-scale FFT representations
- Designed to expose spectral inconsistencies caused by synthesis and resampling

### 3. Temporal Model (VGG16 + LSTM)
- Models temporal inconsistencies across consecutive video frames
- Combines a VGG16 model with an LSTM sequence model
- Produces sequence-level predictions

### 4. Transformer Model (ViT-B/16)
- Uses patch-based self-attention to capture global inconsistencies
- Vision Transformer with 16×16 patch size
- Fine-tuned for binary deepfake classification

---

## Datasets

### FaceForensics++ (FF++, c23)
- Used **only for the temporal model**
- Contains real and manipulated videos generated using multiple deepfake techniques
- Videos are H.264 compressed (CRF 23)
- Video-level splitting is used to avoid identity and frame leakage

### OpenForensics (Image Dataset)
- Used for **spatial, frequency-based, and transformer-based models**
- Contains real and manipulated face images from diverse synthesis pipelines
- Images are resized and cropped to 224×224
- Balanced splits:
  - Train: 9,000 real / 9,000 fake
  - Validation: 3,900 real / 3,900 fake
  - Test: 3,200 real / 3,200 fake

---

## Preprocessing Pipeline

All models use a **shared preprocessing pipeline**:

- Face detection and alignment using **MTCNN**
- Cropping slightly beyond detected facial boundaries
- Resizing to **224 × 224**
- RGB format with pixel values normalized to [0, 1]
- Data augmentation (training only):
  - Random horizontal flips
  - Minor color jittering

This ensures consistency and reduces dataset-induced bias.

---

## Experimental Setup

- **Language:** Python 3.12
- **Framework:** PyTorch 2.0
- **Hardware:** NVIDIA RTX 3060 Ti (16GB RAM)
- **Acceleration:** CUDA
- **Metrics:** Accuracy, test loss, confusion matrix
- **Batch sizes:**
  - ViT: 32
  - Other frame-based models: 16
  - Temporal model: 16 sequences (each with 16 consecutive frames)
- **Early stopping** based on validation loss
- **Evaluation tools:** scikit-learn

---

## Quantitative Results (Summary)

| Method | Dataset | Test Accuracy |
|------|--------|--------------|
| Spatial CNN (EfficientNet-B4) | OpenForensics | 92.11% |
| Frequency CNN (FreqNet) | OpenForensics | 83.31% |
| Temporal (VGG16 + LSTM) | FaceForensics++ | 86.68% |
| Transformer (ViT-B/16) | OpenForensics | **97.86%** |



---

## Reproducibility

- Fixed random seeds where applicable
- Class-balanced dataset splits
- Video-level splitting for temporal models
- All preprocessing, training, and evaluation code included

---

## Author

**Abd-Ul-Haq Amine Ladrem**  
University of Basel
