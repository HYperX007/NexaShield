# NexaShield AI — Model Notes

## Purpose

NexaShield explores machine-learning approaches for identifying synthetic or AI-generated speech.

## Current Approach

- 16 kHz mono audio
- Fixed 3-second analysis windows
- 128-bin Mel-spectrograms
- PyTorch convolutional classifier
- Multi-window probability aggregation
- Separate application risk engine

## Classes

The classifier produces probabilities for:

- Genuine
- Synthetic

The application layer converts these signals into an overall verdict and risk information.

## Evaluation

Model results should always be interpreted in the context of the dataset and split used. Accuracy on a held-out dataset does not establish performance on every unseen TTS or voice-conversion system.

Future evaluation should report:

- Precision / recall
- F1 score
- ROC-AUC
- Confusion matrix
- Speaker-independent splits
- Generator-independent tests
- False positives and false negatives
- Confidence calibration

## Limitations

Performance can change with:

- New synthesis methods
- Different speakers or languages
- Phone microphones
- Audio compression
- Reverberation
- Background noise
- Very short recordings
- Distribution shift

## Responsible Use

NexaShield should be treated as a **risk-analysis and research prototype**, not as definitive proof that an audio recording is genuine or synthetic.

A high synthetic score does not by itself prove impersonation or fraud, and a low score does not prove authenticity.

## Research Direction

The project is being developed toward stronger generalization, robustness, calibrated uncertainty, and evaluation against previously unseen synthesis methods.
