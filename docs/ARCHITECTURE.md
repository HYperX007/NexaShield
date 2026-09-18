# NexaShield AI — Architecture

## High-Level Flow

```text
React + Vite UI
      |
      | HTTP
      v
FastAPI Backend
      |
      +--> FFmpeg audio conversion
      |
      +--> Voice activity check
      |
      v
Multi-window detector
      |
      v
Mel-spectrogram features
      |
      v
PyTorch Voice CNN
      |
      v
Synthetic probability scores
      |
      v
Risk engine
      |
      v
Verdict + risk + recommendation
```

## Audio Pipeline

Incoming audio is converted to a consistent working format:

- Mono
- 16 kHz sample rate
- WAV

The backend performs a lightweight energy-based voice-activity check before classification. Recordings with insufficient speech activity are not blindly classified.

## ML Inference

The current model pipeline is:

```text
Waveform
  -> normalization
  -> fixed-duration window
  -> Mel-spectrogram
  -> power-to-dB conversion
  -> PyTorch tensor
  -> VoiceCNNv3
  -> softmax probabilities
```

The CNN uses convolutional layers, batch normalization, ReLU activations, pooling, adaptive average pooling, and a fully connected classifier.

## Multi-Window Analysis

The backend can analyze multiple audio windows and aggregate their synthetic probabilities.

It reports:

- Average synthetic probability
- Maximum synthetic probability
- Suspicious-window count
- Total analyzed windows

These signals are passed to the separate risk engine.

## Risk Layer

The ML probability and the application decision are deliberately separated:

```text
ML probability != final application decision
```

This makes thresholds easier to tune and keeps the system's decision logic explainable and replaceable.

## Components

- **Frontend:** React + Vite
- **Backend:** Python + FastAPI + Uvicorn
- **Audio:** FFmpeg + Librosa
- **ML:** PyTorch + CNN + Mel-spectrogram features
- **Runtime:** HTTP API between frontend and backend

## Future Directions

- Speaker-independent evaluation
- Generator-independent benchmarks
- Better confidence calibration
- Stronger audio augmentation
- Pretrained speech representations
- Lower-latency inference
- More systematic false-positive and false-negative analysis
