from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

import librosa
import numpy as np
import torch
import torch.nn as nn


SAMPLE_RATE = 16000
WINDOW_SECONDS = 3
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
N_MELS = 128


# =========================
# V3 CNN
# =========================

class VoiceCNNv3(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# =========================
# Load model
# =========================

model = VoiceCNNv3()

model.load_state_dict(
    torch.load(
        BASE_DIR / "data/processed/v2/nexashield_cnn_v3.pth",
        map_location="cpu"
    )
)

model.eval()


# =========================
# Convert one window to Mel
# =========================

def window_to_mel(audio):

    if len(audio) < WINDOW_SAMPLES:

        audio = np.pad(
            audio,
            (0, WINDOW_SAMPLES - len(audio))
        )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


# =========================
# Analyze complete recording
# =========================

def analyze_file(file_path):

    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = librosa.util.normalize(audio)

    duration = len(audio) / SAMPLE_RATE

    # For short files, analyze one padded window
    if len(audio) <= WINDOW_SAMPLES:

        windows = [audio]

    else:

        windows = []

        for start in range(
            0,
            len(audio) - WINDOW_SAMPLES + 1,
            WINDOW_SAMPLES
        ):

            window = audio[
                start:start + WINDOW_SAMPLES
            ]

            windows.append(window)


        # Analyze remaining tail if there is enough audio
        remainder = len(audio) % WINDOW_SAMPLES

        if remainder > SAMPLE_RATE:

            windows.append(
                audio[-WINDOW_SAMPLES:]
            )


    synthetic_scores = []


    for window in windows:

        mel = window_to_mel(window)

        tensor = torch.tensor(
            mel,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)


        with torch.no_grad():

            output = model(tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )


        synthetic_probability = (
            probabilities[0][1].item()
        )

        synthetic_scores.append(
            synthetic_probability
        )


    # Aggregate multiple windows
    average_score = np.mean(
        synthetic_scores
    )

    maximum_score = np.max(
        synthetic_scores
    )


    # Main decision uses average evidence
    if average_score < 0.30:

        result = "GENUINE"

    elif average_score < 0.70:

        result = "UNCERTAIN"

    else:

        result = "SUSPICIOUS"


    return (
        duration,
        len(windows),
        average_score * 100,
        maximum_score * 100,
        result
    )


# =========================
# Test previous recordings
# =========================

TEST_DIR = BASE_DIR / "data/test/real_samples"

files = []

for extension in [
    "*.wav",
    "*.flac",
    "*.mp3",
    "*.mpeg"
]:

    files.extend(
        TEST_DIR.glob(extension)
    )

files = sorted(files)


print("============================================================")
print("NexaShield V3 - Multi-Window Real Voice Test")
print("============================================================")

print()

print(
    f"{'File':<25}"
    f"{'Duration':>10}"
    f"{'Windows':>9}"
    f"{'Avg Synth':>12}"
    f"{'Max Synth':>12}"
    f"{'Result':>14}"
)

print("-" * 85)


for file_path in files:

    try:

        duration, windows, average, maximum, result = (
            analyze_file(file_path)
        )

        print(
            f"{file_path.name:<25}"
            f"{duration:>9.2f}s"
            f"{windows:>9}"
            f"{average:>11.2f}%"
            f"{maximum:>11.2f}%"
            f"{result:>14}"
        )

    except Exception as error:

        print(
            f"ERROR: {file_path.name} -> {error}"
        )


# =========================
# Test completely unseen
# =========================

UNSEEN_DIR = BASE_DIR / "data/raw/additional_real/unseen"

unseen_files = sorted(
    UNSEEN_DIR.glob("*.wav")
)


print()
print("============================================================")
print("Completely Unseen Real Recordings")
print("============================================================")

print()

print(
    f"{'File':<25}"
    f"{'Duration':>10}"
    f"{'Windows':>9}"
    f"{'Avg Synth':>12}"
    f"{'Max Synth':>12}"
    f"{'Result':>14}"
)

print("-" * 85)


for file_path in unseen_files:

    try:

        duration, windows, average, maximum, result = (
            analyze_file(file_path)
        )

        print(
            f"{file_path.name:<25}"
            f"{duration:>9.2f}s"
            f"{windows:>9}"
            f"{average:>11.2f}%"
            f"{maximum:>11.2f}%"
            f"{result:>14}"
        )

    except Exception as error:

        print(
            f"ERROR: {file_path.name} -> {error}"
        )


print()
print("============================================================")
print("Testing complete")
print("============================================================")