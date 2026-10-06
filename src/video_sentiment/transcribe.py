from __future__ import annotations

import sys
from pathlib import Path

from video_sentiment.config import Settings
from video_sentiment.errors import VideoSentimentError


def transcribe(wav: Path, settings: Settings, single_speaker: bool = False) -> list[dict]:
    from video_sentiment.audio import ensure_ffmpeg_on_path
    from video_sentiment.compat import accept_legacy_hub_token, trust_local_model_checkpoints

    ensure_ffmpeg_on_path()
    trust_local_model_checkpoints()
    try:
        import whisperx
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers
    except ImportError as exc:
        raise VideoSentimentError(
            'Speech models are not installed. From the project folder run: '
            'pip install -e ".[ml]"'
        ) from exc

    device = settings.resolved_device()
    compute_type = settings.compute_type()
    _say(f"Loading {settings.whisper_model} on {device} ({compute_type})")
    audio = whisperx.load_audio(str(wav))
    prompt = settings.asr_prompt()
    load_kwargs = {"asr_options": {"initial_prompt": prompt}} if prompt else {}
    try:
        model = whisperx.load_model(
            settings.whisper_model,
            device,
            compute_type=compute_type,
            language=settings.language,
            **load_kwargs,
        )
    except TypeError:
        model = whisperx.load_model(
            settings.whisper_model,
            device,
            compute_type=compute_type,
            language=settings.language,
        )
    result = model.transcribe(
        audio,
        batch_size=settings.batch_size,
        language=settings.language,
    )
    _say("Aligning words")
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=settings.language,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as exc:
        print(f"Word alignment skipped: {exc}", file=sys.stderr)

    if single_speaker:
        for segment in result["segments"]:
            segment["speaker"] = "SPEAKER_00"
            for word in segment.get("words") or []:
                word["speaker"] = "SPEAKER_00"
        return result["segments"]

    if not settings.hf_token:
        raise VideoSentimentError(
            "HF_TOKEN is missing. Create a Hugging Face read token, accept the terms for "
            "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0, "
            "then set HF_TOKEN. Pass --single-speaker to label the whole video as Speaker 1."
        )

    _say("Identifying speakers")
    accept_legacy_hub_token()
    try:
        model_name = "pyannote/speaker-diarization-3.1"
        try:
            diarize_model = DiarizationPipeline(
                token=settings.hf_token,
                device=device,
                model_name=model_name,
            )
        except TypeError:
            diarize_model = DiarizationPipeline(
                use_auth_token=settings.hf_token,
                device=device,
                model_name=model_name,
            )
    except AttributeError as exc:
        raise VideoSentimentError(
            "Could not load the speaker model. While logged in to Hugging Face, accept the terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1 and "
            "https://huggingface.co/pyannote/segmentation-3.0, then run the command again."
        ) from exc
    num_speakers = None
    if (
        settings.min_speakers is not None
        and settings.min_speakers == settings.max_speakers
    ):
        num_speakers = settings.min_speakers
    diarize_segments = diarize_model(
        audio,
        num_speakers=num_speakers,
        min_speakers=settings.min_speakers,
        max_speakers=settings.max_speakers,
    )
    if isinstance(diarize_segments, tuple):
        diarize_segments = diarize_segments[0]
    result = assign_word_speakers(diarize_segments, result, fill_nearest=True)
    return result["segments"]


def _say(message: str) -> None:
    print(message, flush=True)
