from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# =========================
# Configuration
# =========================

SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 128
TARGET_SAMPLES = SAMPLE_RATE * DURATION

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


# =========================
# Dataset folders
# =========================

REAL_DIR = BASE_DIR / "data/raw/improved/real"

ADDITIONAL_REAL_DIR = (
    BASE_DIR / "data/raw/additional_real/training"
)

SYNTHETIC_DIR = BASE_DIR / "data/raw/improved/synthetic"


# =========================
# Collect files
# =========================

real_files = (
    list(REAL_DIR.glob("*.wav"))
    + list(REAL_DIR.glob("*.flac"))
)

additional_real_files = (
    list(ADDITIONAL_REAL_DIR.glob("*.wav"))
    + list(ADDITIONAL_REAL_DIR.glob("*.flac"))
)

synthetic_files = (
    list(SYNTHETIC_DIR.glob("*.wav"))
    + list(SYNTHETIC_DIR.glob("*.flac"))
)


all_files = []
all_labels = []


for file_path in real_files:
    all_files.append(file_path)
    all_labels.append(0)


for file_path in additional_real_files:
    all_files.append(file_path)
    all_labels.append(0)


for file_path in synthetic_files:
    all_files.append(file_path)
    all_labels.append(1)


print("================================")
print("NexaShield V3 Dataset")
print("================================")

print("Genuine:", all_labels.count(0))
print("Synthetic:", all_labels.count(1))
print("Total:", len(all_files))


# =========================
# Train / validation / test
# =========================

train_files, temp_files, train_labels, temp_labels = (
    train_test_split(
        all_files,
        all_labels,
        test_size=0.30,
        random_state=42,
        stratify=all_labels
    )
)


val_files, test_files, val_labels, test_labels = (
    train_test_split(
        temp_files,
        temp_labels,
        test_size=0.50,
        random_state=42,
        stratify=temp_labels
    )
)


print()
print("Training files:", len(train_files))
print("Validation files:", len(val_files))
print("Testing files:", len(test_files))


# =========================
# Audio processing
# =========================

def load_audio(file_path, training=False):

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

        if training:
            max_start = len(audio) - TARGET_SAMPLES
            start = np.random.randint(0, max_start + 1)
        else:
            start = 0

        audio = audio[
            start:start + TARGET_SAMPLES
        ]

    return audio

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

    return audio


def audio_to_mel(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db.astype(np.float32)


# =========================
# Audio augmentation
# =========================

def augment_audio(audio):

    augmented = audio.copy()


    # Random volume change
    gain = np.random.uniform(
        0.8,
        1.2
    )

    augmented = augmented * gain


    # Random time shift
    shift = np.random.randint(
        -1600,
        1601
    )

    if shift > 0:

        augmented = np.pad(
            augmented[:-shift],
            (shift, 0)
        )

    elif shift < 0:

        shift = abs(shift)

        augmented = np.pad(
            augmented[shift:],
            (0, shift)
        )


    # Add small background noise
    if np.random.random() < 0.5:

        noise_strength = np.random.uniform(
            0.001,
            0.008
        )

        noise = np.random.normal(
            0,
            noise_strength,
            size=augmented.shape
        )

        augmented = augmented + noise


    # Prevent clipping
    augmented = np.clip(
        augmented,
        -1.0,
        1.0
    )


    return augmented


# =========================
# PyTorch dataset
# =========================

class VoiceDataset(Dataset):

    def __init__(
        self,
        files,
        labels,
        training=False
    ):

        self.files = files
        self.labels = labels
        self.training = training


    def __len__(self):

        return len(self.files)


    def __getitem__(self, index):

        file_path = self.files[index]
        label = self.labels[index]


        audio = load_audio(
            file_path,
            training=self.training
        )


        # Augmentation ONLY during training
        if self.training:

            if np.random.random() < 0.7:

                audio = augment_audio(audio)


        mel = audio_to_mel(audio)


        tensor = torch.tensor(
            mel,
            dtype=torch.float32
        ).unsqueeze(0)


        return tensor, torch.tensor(
            label,
            dtype=torch.long
        )


# =========================
# Create datasets
# =========================

train_dataset = VoiceDataset(
    train_files,
    train_labels,
    training=True
)

val_dataset = VoiceDataset(
    val_files,
    val_labels,
    training=False
)

test_dataset = VoiceDataset(
    test_files,
    test_labels,
    training=False
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE
)


# =========================
# V3 CNN
# =========================

class VoiceCNNv3(nn.Module):

    def __init__(self):

        super().__init__()


        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),


            nn.AdaptiveAvgPool2d((4, 4))
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 4 * 4,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                2
            )
        )


    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


model = VoiceCNNv3()


# =========================
# Training setup
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# =========================
# Training
# =========================

print()
print("================================")
print("Starting V3 training")
print("================================")


best_val_accuracy = 0.0
best_state = None


for epoch in range(EPOCHS):


    # -------------------------
    # Training
    # -------------------------

    model.train()

    total_loss = 0
    correct = 0
    total = 0


    for batch_X, batch_y in train_loader:

        optimizer.zero_grad()


        outputs = model(batch_X)


        loss = criterion(
            outputs,
            batch_y
        )


        loss.backward()

        optimizer.step()


        total_loss += loss.item()


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == batch_y
        ).sum().item()


        total += batch_y.size(0)


    train_accuracy = correct / total


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            outputs = model(batch_X)

            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_correct += (
                predictions == batch_y
            ).sum().item()


            val_total += batch_y.size(0)


    val_accuracy = val_correct / val_total


    # Save best model

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_state = {
            key: value.cpu().clone()
            for key, value in model.state_dict().items()
        }


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"- Loss: {total_loss:.4f} "
        f"- Train: {train_accuracy * 100:.2f}% "
        f"- Val: {val_accuracy * 100:.2f}%"
    )


# =========================
# Restore best model
# =========================

model.load_state_dict(
    best_state
)


# =========================
# Final test
# =========================

model.eval()

test_correct = 0
test_total = 0


with torch.no_grad():

    for batch_X, batch_y in test_loader:

        outputs = model(batch_X)

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        test_correct += (
            predictions == batch_y
        ).sum().item()


        test_total += batch_y.size(0)


test_accuracy = test_correct / test_total


# =========================
# Save V3
# =========================

MODEL_PATH = (
    BASE_DIR / "data/processed/v2/nexashield_cnn_v3.pth"
)


torch.save(
    model.state_dict(),
    MODEL_PATH
)


print()
print("================================")
print("V3 Training Complete")
print("================================")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Final test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print()
print("V3 model saved to:")
print(MODEL_PATH)