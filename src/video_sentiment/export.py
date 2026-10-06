from __future__ import annotations

import csv
import json
from pathlib import Path

from video_sentiment.sentences import Sentence, Target

LABELS = ("positive", "negative", "neutral", "mixed")

UTTERANCE_FIELDS = [
    "source",
    "speaker",
    "start",
    "end",
    "text",
    "overlap",
    "surface",
    "resolved",
    "resolution",
    "sentiment",
    "evidence",
    "text_confidence",
    "tone_valence",
    "tone_polarity",
    "tone_conflict",
    "sentiment_confidence",
]

SUMMARY_FIELDS = [
    "source",
    "speaker",
    "target",
    "positive",
    "negative",
    "neutral",
    "mixed",
    "dominant",
    "mentions",
]


def dominant(counts: dict[str, int]) -> str:
    best = max(counts[label] for label in LABELS)
    if best == 0:
        return "neutral"
    winners = [label for label in LABELS if counts[label] == best]
    if "positive" in winners and "negative" in winners:
        return "mixed"
    if len(winners) == 1:
        return winners[0]
    return "tie"


def build_summary(sentences: list[Sentence]) -> dict:
    buckets: dict[tuple[str, str], dict] = {}
    for sentence in sentences:
        for target in sentence.targets:
            display = " ".join(target.resolved.split())
            slot = buckets.setdefault(
                (sentence.speaker, display.casefold()),
                {
                    "target": display,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "mixed": 0,
                },
            )
            slot[target.sentiment] += 1
    by_speaker: dict[str, dict] = {}
    for (speaker, _), slot in buckets.items():
        counts = {label: slot[label] for label in LABELS}
        by_speaker.setdefault(speaker, {})[slot["target"]] = {
            **counts,
            "dominant": dominant(counts),
        }
    return {"by_speaker": by_speaker}


def build_document(source: str, duration: float, sentences: list[Sentence]) -> dict:
    summary = build_summary(sentences)
    speakers = list(dict.fromkeys(sentence.speaker for sentence in sentences))
    return {
        "source": source,
        "duration_sec": _round(duration, 3),
        "speakers": speakers,
        "utterances": [_sentence_dict(sentence) for sentence in sentences],
        "summary": summary,
    }


def write_outputs(document: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(out_dir / "utterances.csv", UTTERANCE_FIELDS, _utterance_rows(document))
    _write_csv(out_dir / "summary.csv", SUMMARY_FIELDS, _summary_rows(document))


def _sentence_dict(sentence: Sentence) -> dict:
    return {
        "speaker": sentence.speaker,
        "start": _round(sentence.start, 3),
        "end": _round(sentence.end, 3),
        "text": sentence.text,
        "overlap": sentence.overlap,
        "tone_valence": _round(sentence.tone_valence, 3),
        "tone_polarity": sentence.tone_polarity,
        "targets": [_target_dict(target) for target in sentence.targets],
    }


def _target_dict(target: Target) -> dict:
    return {
        "surface": target.surface,
        "resolved": target.resolved,
        "resolution": target.resolution,
        "sentiment": target.sentiment,
        "evidence": target.evidence,
        "text_confidence": _round(target.text_confidence, 4),
        "tone_polarity": target.tone_polarity,
        "tone_valence": _round(target.tone_valence, 3),
        "tone_conflict": target.tone_conflict,
        "sentiment_confidence": _round(target.sentiment_confidence, 4),
    }


def _utterance_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    for utterance in document["utterances"]:
        targets = utterance["targets"] or [None]
        for target in targets:
            rows.append(
                {
                    "source": document["source"],
                    "speaker": utterance["speaker"],
                    "start": utterance["start"],
                    "end": utterance["end"],
                    "text": utterance["text"],
                    "overlap": utterance["overlap"],
                    "surface": "" if target is None else target["surface"],
                    "resolved": "" if target is None else target["resolved"],
                    "resolution": "" if target is None else target["resolution"],
                    "sentiment": "" if target is None else target["sentiment"],
                    "evidence": "" if target is None else target["evidence"],
                    "text_confidence": "" if target is None else target["text_confidence"],
                    "tone_valence": utterance["tone_valence"] if utterance["tone_valence"] is not None else "",
                    "tone_polarity": utterance["tone_polarity"],
                    "tone_conflict": "" if target is None else target["tone_conflict"],
                    "sentiment_confidence": "" if target is None else target["sentiment_confidence"],
                }
            )
    return rows


def _summary_rows(document: dict) -> list[dict]:
    rows: list[dict] = []
    for speaker, targets in document["summary"]["by_speaker"].items():
        for target, counts in targets.items():
            mentions = sum(counts[label] for label in LABELS)
            rows.append(
                {
                    "source": document["source"],
                    "speaker": speaker,
                    "target": target,
                    "positive": counts["positive"],
                    "negative": counts["negative"],
                    "neutral": counts["neutral"],
                    "mixed": counts["mixed"],
                    "dominant": counts["dominant"],
                    "mentions": mentions,
                }
            )
    return rows


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round(value: float | None, places: int) -> float | None:
    if value is None:
        return None
    return round(float(value), places)
