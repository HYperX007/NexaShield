from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn


SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 128
TARGET_SAMPLES = SAMPLE_RATE * DURATION


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


# Load V3 model
model = VoiceCNNv3()

model.load_state_dict(
    torch.load(
        "../data/processed/v2/nexashield_cnn_v3.pth",
        map_location="cpu"
    )
)

model.eval()


def prepare_audio(file_path):

    audio, _ = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = librosa.util.normalize(audio)

    if len(audio) < TARGET_SAMPLES:

        audio = np.pad(
            audio,
            (0, TARGET_SAMPLES - len(audio))
        )

    else:

        audio = audio[:TARGET_SAMPLES]

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


def predict(file_path):

    mel = prepare_audio(file_path)

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

    genuine = probabilities[0][0].item() * 100
    synthetic = probabilities[0][1].item() * 100

    if synthetic < 30:
        result = "GENUINE"

    elif synthetic < 70:
        result = "UNCERTAIN"

    else:
        result = "SUSPICIOUS"

    return genuine, synthetic, result


# =========================
# Previous 10 real recordings
# =========================

TEST_DIR = Path("../data/test/real_samples")

files = []

for extension in ["*.wav", "*.flac", "*.mp3", "*.mpeg"]:
    files.extend(TEST_DIR.glob(extension))

files = sorted(files)


print("==============================================")
print("NexaShield V3 - Real Voice Test")
print("==============================================")

print()
print("Previous real recordings:", len(files))
print()

print(
    f"{'File':<30}"
    f"{'Genuine':>12}"
    f"{'Synthetic':>12}"
    f"{'Result':>14}"
)

print("-" * 70)


genuine_count = 0
uncertain_count = 0
suspicious_count = 0


for file_path in files:

    try:

        genuine, synthetic, result = predict(file_path)

        print(
            f"{file_path.name:<30}"
            f"{genuine:>11.2f}%"
            f"{synthetic:>11.2f}%"
            f"{result:>14}"
        )

        if result == "GENUINE":
            genuine_count += 1

        elif result == "UNCERTAIN":
            uncertain_count += 1

        else:
            suspicious_count += 1

    except Exception as error:

        print(
            f"ERROR: {file_path.name} -> {error}"
        )


# =========================
# Two completely unseen
# =========================

UNSEEN_DIR = Path(
    "../data/raw/additional_real/unseen"
)

unseen_files = sorted(
    UNSEEN_DIR.glob("*.wav")
)


print()
print("==============================================")
print("Completely Unseen Real Recordings")
print("==============================================")

print()

print(
    f"{'File':<30}"
    f"{'Genuine':>12}"
    f"{'Synthetic':>12}"
    f"{'Result':>14}"
)

print("-" * 70)


for file_path in unseen_files:

    try:

        genuine, synthetic, result = predict(file_path)

        print(
            f"{file_path.name:<30}"
            f"{genuine:>11.2f}%"
            f"{synthetic:>11.2f}%"
            f"{result:>14}"
        )

    except Exception as error:

        print(
            f"ERROR: {file_path.name} -> {error}"
        )


# =========================
# Summary
# =========================

print()
print("==============================================")
print("V3 Real-World Test Summary")
print("==============================================")

print(
    "Previous recordings classified genuine:",
    genuine_count
)

print(
    "Previous recordings uncertain:",
    uncertain_count
)

print(
    "Previous recordings suspicious:",
    suspicious_count
)

print()
print("Testing complete.")
