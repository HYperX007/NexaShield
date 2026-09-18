# 🛡️ NexaShield AI

> AI-powered voice deepfake detection and impersonation-risk analysis.

NexaShield AI is a full-stack research/prototype project that analyzes spoken audio for signals associated with **synthetic or AI-generated speech**. The system combines audio preprocessing, a CNN-based classifier, multi-window analysis, voice-activity checks, and a risk engine to turn model outputs into an interpretable result.

## ✨ What NexaShield Does

The current application is designed around this pipeline:

```text
Audio Upload / Demo Sample
          ↓
   Format Conversion
      (FFmpeg)
          ↓
   Voice Activity Check
      (Librosa)
          ↓
   Multi-Window Analysis
          ↓
 Mel-Spectrogram Features
          ↓
    PyTorch Voice CNN
          ↓
Synthetic Probability Scores
          ↓
       Risk Engine
          ↓
Verdict + Risk + Recommendation
```

The API exposes a FastAPI service with basic health/status endpoints, demo-audio serving, and an `/analyze` endpoint for uploaded audio.

## 🧠 Machine Learning

### Feature extraction

The current model pipeline uses:

- 16 kHz mono audio
- Fixed 3-second analysis windows
- 128-bin Mel-spectrograms
- Log-scaled Mel power features

### Model

The current V3 classifier is a compact convolutional neural network implemented in **PyTorch**:

- 4 convolutional stages
- Batch normalization
- ReLU activations
- Max-pooling
- Adaptive average pooling
- Fully connected classifier
- Dropout regularization

The inference pipeline converts audio into a Mel-spectrogram tensor and returns genuine/synthetic probabilities.

### Robustness-oriented processing

The training workflow has also explored:

- Random temporal cropping
- Volume augmentation
- Time shifting
- Low-level background-noise augmentation
- Stratified train/validation/test splitting

The backend additionally performs a lightweight voice-activity check before AI analysis so silent or extremely low-level recordings are not blindly classified.

## 🏗️ System Architecture

NexaShield is organized into separate frontend, backend, and ML-oriented components:

```text
┌──────────────────────┐
│   React + Vite UI    │
│   Upload / Results   │
└──────────┬───────────┘
           │ HTTP
           ▼
┌──────────────────────┐
│     FastAPI API      │
│ validation + FFmpeg  │
│ voice activity check │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Multi-window Detector│
│  audio-level scoring │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   PyTorch Voice CNN  │
│ Mel → probability    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│      Risk Engine     │
│ score + level +      │
│ recommendation       │
└──────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for more detail.

## 📁 Repository Structure

```text
NexaShield/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── multi_window_detector.py
│   │   └── risk_engine.py
│   ├── demo_samples/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
│
├── data/
│   ├── raw/
│   └── processed/
│
├── ml/
│   └── ...
│
├── docs/
│   └── ARCHITECTURE.md
│
└── README.md
```

> Dataset files, generated NumPy artifacts, local environments, and model artifacts are intentionally excluded or selectively ignored through `.gitignore`.

## 🚀 Running Locally

### 1. Clone

```bash
git clone https://github.com/HYperX007/NexaShield.git
cd NexaShield
```

### 2. Backend

Create and activate a Python environment, then install the backend dependencies:

```bash
cd backend
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The backend expects **FFmpeg** to be available. You can provide an `FFMPEG_PATH` environment variable when a custom path is needed.

From the `backend` directory:

```bash
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend is built with **React + Vite**.

## 🔌 API

### `GET /`

Returns basic project/status information.

### `GET /health`

Health-check endpoint.

### `POST /analyze`

Accepts an uploaded audio file.

Supported input extensions currently include:

```text
.wav  .webm  .mp3  .ogg  .m4a  .flac
```

The response can include:

- genuine probability
- synthetic probability
- maximum synthetic probability
- suspicious window count
- total analysis windows
- risk score
- risk level
- decision/recommendation
- overall verdict
- per-window analysis information

## ⚠️ Important Limitations

NexaShield is a **research/prototype system**, not a forensic guarantee of authenticity.

Model performance can change substantially with:

- unseen TTS/voice-conversion systems
- recording devices and microphones
- codecs and compression
- background noise
- language and speaker differences
- adversarial or intentionally manipulated audio
- dataset composition and distribution shift

A classifier probability should therefore be treated as a **risk signal**, not as definitive proof that a voice is genuine or synthetic.

## 🔬 Research Direction

Planned/ongoing directions include:

- stronger speaker-independent evaluation
- more unseen-generation benchmarks
- calibration of risk scores
- improved audio augmentation
- comparison against pretrained speech representations such as wav2vec-style embeddings
- better real-time inference performance
- stronger documentation of false positives and false negatives

## 🧰 Tech Stack

**Frontend**
- React
- Vite
- JavaScript

**Backend**
- Python
- FastAPI
- Uvicorn
- FFmpeg

**Machine Learning**
- PyTorch
- Librosa
- NumPy
- CNN
- Mel-spectrogram features

## 👤 Author

**Tanishque Mondal**

B.Tech — Electronics & Computer Science  
Narula Institute of Technology

- LinkedIn: https://www.linkedin.com/in/tanishque-mondal-9b62b3397/
- GitHub: https://github.com/HYperX007

---

### 📌 Project Status

NexaShield is actively being developed as an experimental voice-security project. The codebase, model pipeline, evaluation strategy, and deployment workflow are evolving.

**Build → Test → Analyze → Improve.**
