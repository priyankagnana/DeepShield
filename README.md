# 🎭 Real-Time Deepfake Detection System

## 📌 Overview

Deepfake technology uses artificial intelligence to manipulate facial expressions and generate highly realistic fake videos. While powerful, this technology poses serious risks including misinformation, identity theft, cybercrime, and political manipulation.

This project presents a Deep Learning–based Deepfake Detection System that classifies videos as:

- ✅ REAL
- ❌ FAKE

The system also provides model explainability using Grad-CAM heatmaps to visualize the regions influencing predictions.

---

# 🏗 System Architecture

## 🔹 High Level Architecture
           ┌────────────────────┐
           │     Input Video     │
           └──────────┬──────────┘
                      ↓
           ┌────────────────────┐
           │  Frame Extraction   │
           └──────────┬──────────┘
                      ↓
           ┌────────────────────┐
           │   Face Detection    │
           └──────────┬──────────┘
                      ↓
           ┌────────────────────┐
           │  Image Preprocessing│
           │ (Resize, Normalize) │
           └──────────┬──────────┘
                      ↓
           ┌────────────────────────────┐
           │  CNN + Frequency Analysis   │
           │  (Spatial + FFT Branch)     │
           └──────────┬──────────────────┘
                      ↓
           ┌────────────────────┐
           │   Classification    │
           │   Real / Fake       │
           └──────────┬──────────┘
                      ↓
           ┌────────────────────┐
           │   Grad-CAM Module   │
           │   Heatmap Output    │
           └────────────────────┘

---

## 🔹 Detailed Pipeline Architecture

### 1. Data Layer
- Deepfake Detection Challenge Dataset
- REAL and FAKE videos
- Metadata-based labeling

### 2. Preprocessing Layer
- Video loading
- Frame extraction (every Nth frame)
- Face cropping
- Resize to 224x224
- Normalization

### 3. Feature Extraction Layer

#### Spatial Branch (CNN)
- Convolution Layers
- Batch Normalization
- ReLU Activation
- Max Pooling
- Fully Connected Layers

#### Frequency Branch
- Fast Fourier Transform (FFT)
- Frequency artifact extraction
- Feature fusion with spatial features

### 4. Classification Layer
- Dense Layer
- Sigmoid Activation
- Binary Output (Real = 0, Fake = 1)

### 5. Explainability Layer
- Grad-CAM
- Heatmap overlay on frames
- Visual focus area highlighting

---

## 🎯 Problem Statement

The rise of deepfake videos has created major security and trust issues across digital platforms. Manual verification is inefficient and unreliable. An automated AI-based detection system is necessary to:

- Detect manipulated facial regions
- Identify frequency inconsistencies
- Provide explainable predictions
- Support real-time inference

---

## 💡 Proposed Solution

This system implements a computer vision pipeline that:

1. Extracts frames from videos
2. Detects faces
3. Preprocesses images
4. Trains a Convolutional Neural Network (CNN)
5. Applies frequency-domain analysis
6. Classifies real vs fake
7. Generates Grad-CAM heatmaps for interpretability

---

## 📂 Project Structure
deepfake-detection-system/
│
├── data/
│ ├── raw/
│ │ ├── real/
│ │ ├── fake/
│ │ └── metadata.json
│ │
│ └── processed/
│ ├── real/
│ └── fake/
│
├── preprocessing/
│ ├── dataset_split.py
│ ├── frame_extractor.py
│ ├── face_detector.py
│ └── augmentations.py
│
├── model/
│ ├── cnn_model.py
│ ├── frequency_branch.py
│ └── loss.py
│
├── training/
│ ├── train.py
│ ├── evaluate.py
│ ├── metrics.py
│ └── early_stopping.py
│
├── inference/
│ ├── predict.py
│ └── realtime_inference.py
│
├── explainability/
│ ├── gradcam.py
│ └── heatmap_utils.py
│
├── notebooks/
│ ├── EDA.ipynb
│ └── FFT_experiments.ipynb
│
├── app.py
├── requirements.txt
└── README.md

---

## 🛠 Tech Stack

### Programming Language
- Python 3.x

### Deep Learning
- PyTorch / TensorFlow

### Computer Vision
- OpenCV
- CNN Architecture
- FFT (Frequency Analysis)

### Data Processing
- NumPy
- Pandas
- Scikit-learn

### Visualization
- Matplotlib
- Seaborn

### Explainability
- Grad-CAM

### Deployment
- Streamlit

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🔥 Key Features

- Binary classification (Real vs Fake)
- Frame-level deepfake detection
- Spatial + Frequency feature fusion
- Model interpretability via Grad-CAM
- Real-time inference capability

---

## 🚀 Applications

- Social media content verification
- News authenticity validation
- Cybercrime detection
- Digital identity protection
- Media forensics

---

![ER Diagram](assets/Real_Time_deepfake_Detection.png)

## 🔮 Future Enhancements

- Transformer-based models
- 3D CNN for temporal modeling
- EfficientNet backbone
- Cloud deployment (AWS/GCP)
- Mobile integration


## Role Distribution
| Priyanka | ML Lead (Model + Training Head) |
|----------|----------------------------------|
| Aditi   | System + Backend Engineer |
| Aparajita | Frontend + Visualization Engineer |

