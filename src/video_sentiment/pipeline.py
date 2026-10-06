from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from video_sentiment.config import Settings
from video_sentiment.errors import VideoSentimentError
from video_sentiment.export import build_document, write_outputs
from video_sentiment.fuse import apply_tone
from video_sentiment.sentences import (
    Sentence,
    Target,
    canonical_speakers,
    mark_overlaps,
    segments_to_words,
    split_sentences,
    words_to_sentences,
)
from video_sentiment.sentiment import (
    SYSTEM_PROMPT,
    ollama_complete,
    parse_targets,
    user_message,
)


def analyze_video(
    video: Path,
    out_dir: Path,
    settings: Settings,
    skip_tone: bool = False,
    single_speaker: bool = False,
    source_label: str | None = None,
) -> dict:
    from video_sentiment.audio import ensure_ffmpeg_on_path, extract_wav, wav_duration

    ensure_ffmpeg_on_path()
    from video_sentiment.transcribe import transcribe

    if not video.exists():
        raise VideoSentimentError(f"Video not found: {video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "audio.wav"
    print(f"Extracting audio to {wav}", flush=True)
    extract_wav(video, wav)
    segments = transcribe(wav, settings, single_speaker=single_speaker)
    sentences = canonical_speakers(words_to_sentences(segments_to_words(segments)))
    mark_overlaps(sentences)
    print(f"{len(sentences)} sentences", flush=True)
    if not skip_tone:
        from video_sentiment.tone import tag_sentences

        print("Reading voice tone", flush=True)
        tag_sentences(wav, sentences, settings)
    score_sentences(sentences, settings)
    document = build_document(source_label or str(video), wav_duration(wav), sentences)
    write_outputs(document, out_dir)
    print(f"Wrote {out_dir / 'result.json'}", flush=True)
    return document


def score_transcript(transcript: Path, out_dir: Path, settings: Settings) -> dict:
    sentences = load_turns(transcript)
    mark_overlaps(sentences)
    score_sentences(sentences, settings)
    duration = max((sentence.end for sentence in sentences), default=0.0)
    document = build_document(str(transcript), duration, sentences)
    write_outputs(document, out_dir)
    print(f"Wrote {out_dir / 'result.json'}", flush=True)
    return document


def score_sentences(
    sentences: list[Sentence],
    settings: Settings,
    complete: Callable[[str, str], str] | None = None,
) -> list[Sentence]:
    def default_complete(system: str, user: str) -> str:
        return ollama_complete(settings.ollama_url, settings.ollama_model, system, user)

    complete_fn = complete or default_complete
    window = max(0, settings.context_sentences)
    total = len(sentences)
    for index, sentence in enumerate(sentences):
        history = sentences[max(0, index - window) : index]
        print(f"[{index + 1}/{total}] {sentence.speaker}", flush=True)
        message = user_message(history, sentence)
        raw = complete_fn(SYSTEM_PROMPT, message)
        try:
            drafts = parse_targets(raw)
        except VideoSentimentError:
            raw = complete_fn(
                SYSTEM_PROMPT,
                message + "\n\nYour previous reply was not valid JSON. Return the JSON object only.",
            )
            drafts = parse_targets(raw)
        sentence.targets = [_to_target(draft, sentence, settings) for draft in drafts]
    return sentences


def load_turns(path: Path) -> list[Sentence]:
    if not path.exists():
        raise VideoSentimentError(f"Transcript not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VideoSentimentError(f"{path} is not valid JSON.") from exc
    rows = data.get("utterances") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise VideoSentimentError(f"{path} must contain an utterances list.")
    sentences: list[Sentence] = []
    for row in rows:
        if not isinstance(row, dict) or "text" not in row:
            raise VideoSentimentError(f"{path} has an utterance without text.")
        text = " ".join(str(row["text"]).split())
        if not text:
            continue
        sentences.extend(
            _split_utterance(
                Sentence(
                    speaker=str(row.get("speaker") or "Speaker 1"),
                    start=float(row.get("start", 0.0)),
                    end=float(row.get("end", 0.0)),
                    text=text,
                    overlap=bool(row.get("overlap", False)),
                )
            )
        )
    return canonical_speakers(sentences)


def _split_utterance(sentence: Sentence) -> list[Sentence]:
    parts = split_sentences(sentence.text)
    if len(parts) <= 1:
        return [sentence]
    duration = max(0.0, sentence.end - sentence.start)
    total = sum(len(part) for part in parts) or 1
    cursor = sentence.start
    pieces: list[Sentence] = []
    for index, part in enumerate(parts):
        end = sentence.end if index == len(parts) - 1 else cursor + duration * (len(part) / total)
        pieces.append(
            Sentence(
                speaker=sentence.speaker,
                start=cursor,
                end=end,
                text=part,
                overlap=sentence.overlap,
            )
        )
        cursor = end
    return pieces


def _to_target(draft: dict, sentence: Sentence, settings: Settings) -> Target:
    evidence = draft["evidence"] or sentence.text[:180]
    conflict, confidence = apply_tone(
        draft["sentiment"],
        draft["text_confidence"],
        sentence.tone_polarity,
        settings.conflict_confidence_scale,
    )
    return Target(
        surface=draft["surface"],
        resolved=draft["resolved"],
        resolution=draft["resolution"],
        sentiment=draft["sentiment"],
        evidence=evidence,
        text_confidence=draft["text_confidence"],
        tone_polarity=sentence.tone_polarity,
        tone_valence=sentence.tone_valence,
        tone_conflict=conflict,
        sentiment_confidence=confidence,
    )
