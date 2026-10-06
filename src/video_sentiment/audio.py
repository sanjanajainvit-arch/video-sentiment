from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path

from video_sentiment.errors import VideoSentimentError


def ensure_ffmpeg_on_path() -> str:
    """Make ffmpeg visible to tools that call it by name, such as WhisperX."""
    ffmpeg = find_ffmpeg()
    directory = str(Path(ffmpeg).resolve().parent)
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep)
    if directory not in parts:
        os.environ["PATH"] = directory + os.pathsep + current
    return ffmpeg


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    bundled = _winget_ffmpeg()
    if bundled:
        return bundled
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise VideoSentimentError(
            "ffmpeg is not on PATH. Reinstall the project so imageio-ffmpeg is available."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _winget_ffmpeg() -> str | None:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    root = Path(local) / "Microsoft" / "WinGet" / "Packages"
    if not root.exists():
        return None
    matches = sorted(root.glob("Gyan.FFmpeg*/ffmpeg-*-full_build/bin/ffmpeg.exe"))
    return str(matches[-1]) if matches else None


def extract_wav(video: Path, wav: Path) -> None:
    wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        find_ffmpeg(),
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0 or not wav.exists():
        detail = (completed.stderr or completed.stdout or "ffmpeg failed").strip()
        raise VideoSentimentError(detail[-2000:])


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
    if rate <= 0:
        return 0.0
    return frames / float(rate)


def read_wav(path: Path):
    """Return mono float samples in -1..1 and the sample rate."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise VideoSentimentError(f"{path} must be 16-bit PCM audio.")
    try:
        import numpy as np
    except ImportError as exc:
        raise VideoSentimentError(
            'numpy is not installed. From the project folder run: pip install -e ".[ml]"'
        ) from exc

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate
