# 🛡️ DeepShield — Real-Time Deepfake Detection System

A fully offline, explainable deepfake detection system built on **EfficientNet-B0** with Grad-CAM visual explanations, a polished Streamlit UI, and a FastAPI backend for real-time inference.

---

## 📌 Overview

Deepfake technology uses generative AI to create highly realistic synthetic faces in images and videos. While powerful, it poses serious risks — misinformation, identity fraud, impersonation, and reputational harm.

**DeepShield** addresses this with a privacy-first, fully offline detection pipeline that:

- Classifies images and videos as ✅ **Real** or 🚨 **Fake**
- Provides **confidence scores** and **P(Real) / P(Fake)** probabilities
- Explains decisions visually using **Grad-CAM heatmaps**
- Runs entirely on your machine — **no cloud calls, no data leaves your device**

---
## Drive link 
https://drive.google.com/drive/folders/1kyWpFCtVWmF7qZGzEp4oOrDCnI8K6OAN?role=writer

## 🏗️ System Architecture

```
┌─────────────────────┐
│   Input (Image /    │
│   Video / Webcam)   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   Face Detection    │  ← OpenCV Haar Cascade (face_detector.py)
│   & Frame Sampling  │  ← Frame extractor (frame_extractor.py)
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Image Preprocessing│  ← Resize 224×224, ImageNet normalize
└──────────┬──────────┘
           ↓
┌──────────────────────────────────────┐
│         DeepfakeCNN Model            │
│                                      │
│  EfficientNet-B0 (Spatial Branch)    │  → 1280-dim features
│  +                                   │
│  FrequencyBranch (FFT Spectrum)      │  → 128-dim features  [opt-in]
│                                      │
│  Fused → Linear head → Binary logit  │
└──────────┬───────────────────────────┘
           ↓
┌─────────────────────┐
│  Classification     │  Real / Fake + confidence score
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Grad-CAM Module    │  Visual heatmap over suspicious regions
└─────────────────────┘
```

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **EfficientNet-B0 backbone** | ImageNet-pretrained, two-phase fine-tuning |
| **Frequency-domain analysis** | Optional FFT branch detects GAN grid artefacts |
| **Face detection** | OpenCV Haar cascade — crops to face before inference |
| **Grad-CAM explanations** | Heatmap overlay showing which regions drove the decision |
| **Full video analysis** | Samples N frames evenly, aggregates with majority vote + timeline chart |
| **Live webcam** | `streamlit-webrtc` in the UI + CLI realtime script |
| **FastAPI backend** | REST + WebSocket endpoints for image, video, and frame streaming |
| **Fully offline** | No internet connection required for inference |
| **MPS / CUDA / CPU** | Auto-detects Apple Silicon, NVIDIA GPU, or CPU |

---

## 📂 Project Structure

```
DeepShield/
│
├── api/                        ← FastAPI backend
│   ├── main.py                 ← App entry point, model loaded at startup
│   ├── schemas.py              ← Pydantic response models
│   └── routes/
│       ├── predict.py          ← POST /predict/image, POST /predict/video
│       └── stream.py           ← WS /ws/webcam (real-time frame inference)
│
├── model/
│   ├── cnn_model.py            ← DeepfakeCNN (EfficientNet-B0 + optional FrequencyBranch)
│   ├── frequency_branch.py     ← FFT-based spectral feature extractor
│   └── loss.py
│
├── inference/
│   ├── predict.py              ← load_model, predict, predict_image, predict_video, predict_with_gradcam
│   └── realtime_inference.py   ← CLI webcam / video loop with frame skipping
│
├── training/
│   ├── train.py                ← Two-phase EfficientNet fine-tuning
│   ├── evaluate.py             ← Test-set evaluation with tqdm progress
│   ├── dataset.py              ← DataLoader, balanced subset sampling
│   ├── metrics.py              ← Accuracy, precision, recall, F1, confusion matrix
│   └── early_stopping.py
│
├── preprocessing/
│   ├── face_detector.py        ← detect_and_crop_face() using OpenCV Haar cascade
│   ├── frame_extractor.py      ← Extract 30 frames/video with multiprocessing
│   ├── dataset_split.py        ← Sort raw videos → real/ fake/ using metadata.json
│   ├── split_train_val_test.py ← 70/15/15 split grouped by video ID
│   └── augmentations.py
│
├── explainability/
│   ├── gradcam.py              ← Grad-CAM with forward + backward hooks
│   └── heatmap_utils.py        ← Heatmap colormap overlay
│
├── saved_models/
│   └── best_model.pth          ← Best checkpoint saved during training
│
├── app.py                      ← Streamlit UI (Image / Video / Webcam tabs)
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Deep Learning** | PyTorch, TorchVision |
| **Model** | EfficientNet-B0 (ImageNet pretrained) |
| **Computer Vision** | OpenCV |
| **Frequency Analysis** | PyTorch FFT (`torch.fft.fft2`, fftshift) |
| **Explainability** | Grad-CAM (backward hooks) |
| **Frontend** | Streamlit, streamlit-webrtc, Plotly |
| **Backend API** | FastAPI, Uvicorn, WebSockets |
| **Data / Metrics** | NumPy, Pandas, Scikit-learn |
| **Training utilities** | tqdm, early stopping |

---

## 📊 Model Performance

Trained on the **140k Real vs Fake Faces** dataset (Kaggle):

| Metric | Score |
|---|---|
| Accuracy | ~91–92% |
| Precision | — |
| Recall | — |
| F1 Score | — |

> Run `python -m training.evaluate` after training to get exact numbers on your test split.

---

## 📥 Dataset Setup

This project uses the **140k Real vs Fake Faces** dataset from Kaggle.

**Download link:** [https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

After downloading, place it at the project root:

```
DeepShield/
└── 140k-faces/
    └── real_vs_fake/
        └── real-vs-fake/
            ├── train/
            │   ├── real/
            │   └── fake/
            ├── valid/
            │   ├── real/
            │   └── fake/
            └── test/
                ├── real/
                └── fake/
```

> The `140k-faces/` folder is in `.gitignore` and must be placed manually on each machine.

---

## 🚀 Full Setup & Workflow

### Prerequisites

#### 1. System libraries (macOS — install before creating the venv)

```bash
brew install xz cmake libomp
```

#### 2. Python version (3.10+ recommended)

```bash
pyenv install 3.12.2
pyenv local 3.12.2
```

#### 3. Virtual environment

```bash
python3 -m venv venv
source venv/bin/activate       # macOS / Linux
# venv\Scripts\activate        # Windows
```

#### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 1 — Train the Model

```bash
python -m training.train
```

Trains DeepfakeCNN using two-phase EfficientNet fine-tuning for up to 50 epochs. Best checkpoint is saved to `saved_models/best_model.pth` whenever validation accuracy improves.

**Training phases:**
- **Phase 1** (epochs 1–5): Backbone frozen, only the classifier head trains at lr=1e-4
- **Phase 2** (epoch 6+): Last two EfficientNet blocks unfrozen, full model trains at lr=1e-5

Sample output:
```
Epoch 1/50 [Ph1]  train_loss=0.512  val_loss=0.431  val_acc=0.8120
...
Epoch 20/50 [Ph2]  train_loss=0.214  val_loss=0.198  val_acc=0.9167
```

---

### Step 2 — Evaluate on the Test Set

```bash
python -m training.evaluate
```

Loads `saved_models/best_model.pth` and reports Accuracy, Precision, Recall, F1, and Confusion Matrix on the held-out test set. Includes a tqdm progress bar.

Sample output:
```
Evaluating: 100%|████████████| 625/625 [05:23<00:00]

Test set evaluation
----------------------------------------
Accuracy:  0.9167
Precision: 0.9210
Recall:    0.9140
F1:        0.9175
```

---

### Step 3 — Launch the Streamlit App

```bash
streamlit run app.py
```

Opens the full UI at `http://localhost:8501`. Three tabs:

#### 📷 Image Tab
- Upload any face image (JPG/PNG)
- Shows verdict card with confidence %, P(Real), P(Fake)
- Enable **Grad-CAM** in sidebar to see which facial regions influenced the decision
- Plotly donut chart shows Real/Fake probability split

#### 🎬 Video Tab
- Upload a video (MP4/AVI/MOV)
- Choose how many frames to analyze (4–32)
- Summary metrics: frames analyzed, avg P(Real), real/fake frame counts
- Interactive **P(Real) timeline chart** (per-frame line chart with 0.5 threshold)
- **Frame distribution histogram** showing score spread
- Collapsible per-frame detail table

#### 📹 Webcam Tab
- Live webcam feed via `streamlit-webrtc`
- Inference every 3rd frame to keep stream smooth
- Bottom banner shows Real/Fake label + confidence
- Top bar shows P(Real) as a fill indicator
- Falls back gracefully if `streamlit-webrtc` is not installed

**Sidebar options:**
- Toggle Grad-CAM overlay
- Score interpretation table (what P(Real) ranges mean)

---

### Step 4 — Run the FastAPI Backend

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Model is loaded **once at startup** and reused for all requests.

Interactive API docs: `http://localhost:8000/docs`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check if model is loaded and which device is in use |
| `/predict/image` | POST | Upload image → `{label, confidence, prob_real}` |
| `/predict/video` | POST | Upload video → aggregated + per-frame results |
| `/ws/webcam` | WebSocket | Send JPEG bytes → receive JSON predictions in real-time |

Example request (image):
```bash
curl -X POST http://localhost:8000/predict/image \
  -F "file=@face.jpg"
```

Example response:
```json
{
  "label": "Fake",
  "confidence": 0.9312,
  "prob_real": 0.0688
}
```

---

### Step 5 — CLI Real-Time Inference (Webcam or Video)

```bash
# Webcam
python -m inference.realtime_inference

# Video file
python -m inference.realtime_inference --video path/to/video.mp4

# With Grad-CAM overlay
python -m inference.realtime_inference --video path/to/video.mp4 --gradcam
```

Press **Q** to quit. Inference runs every 3rd frame for smooth display.

---

### Quick Reference

```bash
# ── Environment ─────────────────────────────────────────
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# ── Train ───────────────────────────────────────────────
python -m training.train

# ── Evaluate ────────────────────────────────────────────
python -m training.evaluate

# ── Streamlit UI ────────────────────────────────────────
streamlit run app.py

# ── FastAPI backend ─────────────────────────────────────
uvicorn api.main:app --reload --port 8000

# ── CLI webcam / video ───────────────────────────────────
python -m inference.realtime_inference [--video <path>] [--gradcam]
```

---

## 🔬 Model Architecture Details

### DeepfakeCNN

```
EfficientNet-B0 (pretrained on ImageNet)
  └── features[0..8]  (MBConv blocks)
  └── classifier
        ├── Dropout(0.4)
        └── Linear(1280 → 1)          # Default mode

Optional: use_frequency=True
  EfficientNet features (1280-dim)
  + FrequencyBranch (128-dim)
  → Linear(1408 → 256) → ReLU → Dropout(0.4) → Linear(256 → 1)
```

### FrequencyBranch

Detects spectral artefacts characteristic of GAN-generated images:

1. `torch.fft.fft2` — 2D Fast Fourier Transform
2. `fftshift` — centres low-frequency content for spatially-consistent conv filters
3. `log1p` — compresses extreme dynamic range of FFT magnitudes
4. Two Conv2D + BatchNorm + MaxPool blocks
5. Fully connected → 128-dim feature vector

### Two-Phase Training

| Phase | Epochs | LR | Backbone |
|---|---|---|---|
| Phase 1 (warm-up) | 1–5 | 1e-4 | Fully frozen |
| Phase 2 (fine-tune) | 6+ | 1e-5 | Last 2 blocks unfrozen |

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall correct classifications |
| Precision | Of predicted fakes, how many were actually fake |
| Recall | Of actual fakes, how many were caught |
| F1 Score | Harmonic mean of precision and recall |
| Confusion Matrix | True/False Positive/Negative breakdown |

---

## 🌐 Applications

- Social media content verification
- News authenticity validation
- Digital identity protection
- Cybercrime and fraud detection
- Media forensics and journalism

---

## 🔮 Future Enhancements

- Temporal modeling with 3D CNN or Vision Transformer across video frames
- Audio-visual consistency check (voice + face sync)
- Browser extension for in-page detection
- Mobile deployment (CoreML / TFLite)
- Confidence calibration and uncertainty estimation

---

## 📜 License

This project is released under the MIT License.
