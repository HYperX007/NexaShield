import os

import librosa
import numpy as np
import torch

from app.detector import model


SAMPLE_RATE = 16000
N_MELS = 128

WINDOW_SECONDS = 3
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

STEP_SECONDS = 1.5
STEP_SAMPLES = int(SAMPLE_RATE * STEP_SECONDS)


def preprocess_audio(audio):

    audio = librosa.util.normalize(audio)

    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(
            audio,
            (0, WINDOW_SAMPLES - len(audio))
        )
    else:
        audio = audio[:WINDOW_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    tensor = torch.tensor(
        mel_db,
        dtype=torch.float32
    )

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


def analyze_audio_windows(audio_path):

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file contains no usable audio.")

    windows = []

    if len(audio) <= WINDOW_SAMPLES:

        windows.append(audio)

    else:

        start = 0

        while start < len(audio):

            end = start + WINDOW_SAMPLES

            window = audio[start:end]

            if len(window) < WINDOW_SAMPLES:

                window = np.pad(
                    window,
                    (0, WINDOW_SAMPLES - len(window))
                )

            windows.append(window)

            if end >= len(audio):
                break

            start += STEP_SAMPLES

    results = []

    with torch.no_grad():

        for index, window in enumerate(windows):

            tensor = preprocess_audio(window)

            output = model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

            genuine = float(probabilities[0])
            synthetic = float(probabilities[1])

            results.append({
                "window": index + 1,
                "genuine_probability": genuine,
                "synthetic_probability": synthetic
            })

    synthetic_scores = [
        result["synthetic_probability"]
        for result in results
    ]

    suspicious_windows = sum(
        score >= 0.70
        for score in synthetic_scores
    )

    average_synthetic = float(
        np.mean(synthetic_scores)
    )

    maximum_synthetic = float(
        np.max(synthetic_scores)
    )

    return {
        "average_synthetic_probability": average_synthetic,
        "maximum_synthetic_probability": maximum_synthetic,
        "suspicious_windows": suspicious_windows,
        "total_windows": len(results),
        "windows": results
    }