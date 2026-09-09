import os
import shutil
import subprocess
import tempfile

import librosa
import numpy as np
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.multi_window_detector import analyze_audio_windows
from app.risk_engine import calculate_risk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, "..", ".env"))


app = FastAPI(
    title="NexaShield AI",
    description="AI-powered voice cloning detection and impersonation prevention",
    version="0.5.0"
)


# =========================================
# CORS
# =========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://nexashield-ai.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================
# FFMPEG CONFIGURATION
# =========================================

FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")

if not os.path.isabs(FFMPEG_PATH):
    FFMPEG_PATH = shutil.which(FFMPEG_PATH) or FFMPEG_PATH

if not os.path.exists(FFMPEG_PATH) and FFMPEG_PATH != "ffmpeg":
    raise RuntimeError(
        "FFMPEG_PATH is not configured correctly."
    )

# =========================================
# PROJECT PATH
# =========================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


DEMO_AUDIO_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "demo_samples"
)


# =========================================
# BASIC ENDPOINTS
# =========================================

@app.get("/")
def root():
    return {
        "project": "NexaShield AI",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# =========================================
# DEMO AUDIO
# =========================================

@app.get("/demo/{filename}")
def get_demo_audio(filename: str):

    allowed_demo_files = {
        "real_test_01.wav",
        "real_test_02.wav",
        "real_test_06.wav",
        "real_test_07.wav",
        "real_test_08.wav",
        "real_test_09.wav",
    }

    if filename not in allowed_demo_files:
        raise HTTPException(
            status_code=404,
            detail="Demo audio file not found."
        )

    audio_path = os.path.join(
        DEMO_AUDIO_DIR,
        filename
    )

    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=404,
            detail="Demo audio file is not available."
        )

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=filename
    )


# =========================================
# VOICE ACTIVITY CHECK
# =========================================

def detect_voice_activity(wav_path):
    """
    Lightweight voice/activity check.

    This is an energy-based speech presence check.
    It is used before the CNN so that completely silent
    or extremely low-level recordings are not classified
    as genuine/synthetic.
    """

    try:
        audio, sample_rate = librosa.load(
            wav_path,
            sr=16000,
            mono=True
        )

        if audio.size == 0:
            return False

        # Remove DC offset.
        audio = audio - np.mean(audio)

        # Calculate RMS energy over short frames.
        frame_rms = librosa.feature.rms(
            y=audio,
            frame_length=1024,
            hop_length=512
        )[0]

        if frame_rms.size == 0:
            return False

        # Convert RMS energy to dBFS.
        frame_db = 20 * np.log10(
            frame_rms + 1e-10
        )

        max_db = float(np.max(frame_db))

        # Frames above this level are treated as active audio.
        activity_threshold_db = -40.0

        active_frames = frame_db > activity_threshold_db

        active_ratio = float(
            np.mean(active_frames)
        )

        # Minimum amount of meaningful audio activity.
        minimum_active_ratio = 0.05

        # Extremely quiet recordings are rejected.
        minimum_peak_db = -45.0

        if max_db < minimum_peak_db:
            return False

        if active_ratio < minimum_active_ratio:
            return False

        return True

    except Exception as error:
        print("Voice activity check error:", error)

        # If the activity check itself fails,
        # don't silently classify the audio.
        raise


# =========================================
# AUDIO ANALYSIS
# =========================================

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):

    # -----------------------------------------
    # Validate filename
    # -----------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No audio file was provided."
        )


    # -----------------------------------------
    # Validate extension
    # -----------------------------------------

    allowed_extensions = {
        ".wav",
        ".webm",
        ".mp3",
        ".ogg",
        ".m4a",
        ".flac"
    }

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format."
        )


    # -----------------------------------------
    # Temporary processing directory
    # -----------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:

        input_path = os.path.join(
            temp_dir,
            "input_audio" + extension
        )

        wav_path = os.path.join(
            temp_dir,
            "converted.wav"
        )


        # -----------------------------------------
        # Save uploaded audio
        # -----------------------------------------

        try:

            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(
                    file.file,
                    buffer
                )

        except Exception:

            raise HTTPException(
                status_code=400,
                detail="Unable to save the uploaded audio."
            )


        # -----------------------------------------
        # Convert to 16 kHz mono WAV
        # -----------------------------------------

        try:

            subprocess.run(
                [
                    FFMPEG_PATH,
                    "-y",
                    "-i",
                    input_path,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    wav_path
                ],
                check=True,
                capture_output=True,
                text=True
            )

        except subprocess.CalledProcessError:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded audio could not be processed. "
                    "Please try another audio file."
                )
            )


        # =========================================
        # VOICE ACTIVITY CHECK
        # =========================================

        try:

            voice_detected = detect_voice_activity(
                wav_path
            )

        except Exception:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The audio could not be checked for "
                    "voice activity. Please try another recording."
                )
            )


        # -----------------------------------------
        # No voice detected
        # -----------------------------------------

        if not voice_detected:

            return {
                "status": "NO_VOICE_DETECTED",
                "voice_detected": False,
                "filename": file.filename,
                "message": (
                    "NexaShield could not identify sufficient "
                    "speech activity in this recording. "
                    "Please record or upload a voice sample "
                    "containing speech."
                )
            }


        # =========================================
        # MULTI-WINDOW AI ANALYSIS
        # =========================================

        try:

            analysis = analyze_audio_windows(
                wav_path
            )

        except Exception as error:

            print(
                "Detector error:",
                error
            )

            raise HTTPException(
                status_code=400,
                detail=(
                    "The audio could not be analyzed. "
                    "Please try a clearer audio recording."
                )
            )


    # =========================================
    # ANALYSIS RESULTS
    # =========================================

    average_synthetic = analysis[
        "average_synthetic_probability"
    ]

    maximum_synthetic = analysis[
        "maximum_synthetic_probability"
    ]

    suspicious_windows = analysis[
        "suspicious_windows"
    ]

    total_windows = analysis[
        "total_windows"
    ]


    # =========================================
    # RISK ENGINE
    # =========================================

    risk = calculate_risk(
        average_synthetic_probability=average_synthetic,
        maximum_synthetic_probability=maximum_synthetic,
        suspicious_windows=suspicious_windows,
        total_windows=total_windows
    )


    # =========================================
    # VERDICT
    # =========================================

    genuine_probability = 1 - average_synthetic


    if average_synthetic >= 0.70:

        verdict = "SUSPICIOUS"

    elif average_synthetic >= 0.30:

        verdict = "UNCERTAIN"

    else:

        verdict = "GENUINE"


    # =========================================
    # RESPONSE
    # =========================================

    return {

        "status": "ANALYZED",

        "voice_detected": True,

        "filename": file.filename,

        "genuine_probability": round(
            genuine_probability * 100,
            2
        ),

        "synthetic_probability": round(
            average_synthetic * 100,
            2
        ),

        "maximum_synthetic_probability": round(
            maximum_synthetic * 100,
            2
        ),

        "suspicious_windows": suspicious_windows,

        "total_windows": total_windows,

        "risk_score": risk["risk_score"],

        "risk_level": risk["risk_level"],

        "decision": risk["decision"],

        "recommendation": risk["recommendation"],

        "verdict": verdict,

        "windows": analysis["windows"]

    }