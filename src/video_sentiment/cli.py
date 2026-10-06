from __future__ import annotations

import argparse
import sys
from pathlib import Path

from video_sentiment import __version__
from video_sentiment.config import load_settings
from video_sentiment.errors import VideoSentimentError
from video_sentiment.download import default_output_dir, download_audio, is_media_url, safe_folder_name
from video_sentiment.pipeline import analyze_video, score_transcript
from video_sentiment.sentiment import ollama_models


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(run(argv))


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="video-sentiment",
        description="Score what each speaker says about each subject in a discussion video.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Run the full pipeline on a video file or a link.")
    analyze.add_argument("video", help="A video file path, or an http(s) link such as YouTube.")
    analyze.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Folder for results. Defaults to outputs/<YouTube title or video file name>.",
    )
    analyze.add_argument("--config", type=Path)
    analyze.add_argument("--skip-tone", action="store_true")
    analyze.add_argument("--single-speaker", action="store_true")
    analyze.add_argument("--min-speakers", type=int)
    analyze.add_argument("--max-speakers", type=int)

    score = sub.add_parser("score-text", help="Score a JSON transcript. No audio or tone.")
    score.add_argument("transcript", type=Path)
    score.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Folder for results. Defaults to outputs/<transcript file name>.",
    )
    score.add_argument("--config", type=Path)

    check = sub.add_parser("check", help="Show whether local models and tools are ready.")
    check.add_argument("--config", type=Path)

    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.config or _local_config())
        if args.command == "analyze":
            if args.min_speakers is not None:
                settings.min_speakers = args.min_speakers
            if args.max_speakers is not None:
                settings.max_speakers = args.max_speakers
            source = str(args.video).strip()
            source_label = None
            media = Path(source)
            output = args.output or default_output_dir(source)
            print(f"Saving results to {output}", flush=True)
            if is_media_url(source):
                print(f"Downloading {source}", flush=True)
                media = download_audio(source, output)
                source_label = source
            analyze_video(
                media,
                output,
                settings,
                skip_tone=args.skip_tone,
                single_speaker=args.single_speaker,
                source_label=source_label,
            )
        elif args.command == "score-text":
            output = args.output or (Path("outputs") / safe_folder_name(args.transcript.stem))
            print(f"Saving results to {output}", flush=True)
            score_transcript(args.transcript, output, settings)
        else:
            return check_environment(settings)
    except VideoSentimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def check_environment(settings) -> int:
    problems: list[str] = []
    print(f"python {sys.version.split()[0]}")
    try:
        from video_sentiment.audio import find_ffmpeg

        print(f"ffmpeg {find_ffmpeg()}")
    except VideoSentimentError as exc:
        problems.append(str(exc))
        print(f"ffmpeg missing: {exc}")

    try:
        import torch

        print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    except ImportError:
        problems.append("torch is not installed")
        print("torch is not installed")

    try:
        import whisperx

        print(f"whisperx {getattr(whisperx, '__version__', 'installed')}")
    except ImportError:
        problems.append("whisperx is not installed")
        print("whisperx is not installed")

    try:
        import transformers

        print(f"transformers {transformers.__version__}")
    except ImportError:
        problems.append("transformers is not installed")
        print("transformers is not installed")

    if settings.hf_token:
        print("HF_TOKEN is set")
    else:
        problems.append("HF_TOKEN is not set")
        print("HF_TOKEN is not set")

    try:
        models = ollama_models(settings.ollama_url)
        print("ollama models: " + (", ".join(models) if models else "(none)"))
        wanted = settings.ollama_model
        if not any(name == wanted or name.startswith(wanted + ":") for name in models):
            problems.append(f"Ollama model {wanted} is not pulled")
            print(f"missing model {wanted}")
    except VideoSentimentError as exc:
        problems.append(str(exc))
        print(str(exc))

    if problems:
        print(f"{len(problems)} check(s) still needed")
        return 1
    print("ready")
    return 0


def _local_config() -> Path | None:
    candidate = Path("config.yaml")
    if candidate.exists():
        return candidate
    return None


if __name__ == "__main__":
    raise SystemExit(main())
