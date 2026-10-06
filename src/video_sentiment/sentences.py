from __future__ import annotations

import re
from dataclasses import dataclass, field

MIN_OVERLAP_SEC = 0.3

_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc", "ltd", "co"}
_SPEAKER_LABEL = re.compile(r"Speaker \d+")


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: str


@dataclass
class Target:
    surface: str
    resolved: str
    resolution: str
    sentiment: str
    evidence: str
    text_confidence: float
    tone_polarity: str
    tone_valence: float | None
    tone_conflict: bool
    sentiment_confidence: float


@dataclass
class Sentence:
    speaker: str
    start: float
    end: float
    text: str
    overlap: bool = False
    tone_valence: float | None = None
    tone_polarity: str = "unknown"
    targets: list[Target] = field(default_factory=list)


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    merged: list[str] = []
    for part in parts:
        if not part:
            continue
        if merged and _is_abbrev_fragment(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def segments_to_words(segments: list[dict]) -> list[Word]:
    words: list[Word] = []
    for segment in segments:
        speaker = str(segment.get("speaker") or "SPEAKER_00")
        raw_words = segment.get("words") or []
        usable = [
            item
            for item in raw_words
            if "start" in item and "end" in item and str(item.get("word", "")).strip()
        ]
        if usable:
            for item in usable:
                words.append(
                    Word(
                        text=str(item["word"]).strip(),
                        start=float(item["start"]),
                        end=float(item["end"]),
                        speaker=str(item.get("speaker") or speaker),
                    )
                )
            continue
        text = str(segment.get("text", "")).strip()
        if text:
            words.append(
                Word(
                    text=text,
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    speaker=speaker,
                )
            )
    words.sort(key=lambda item: (item.start, item.end))
    return words


def words_to_sentences(words: list[Word]) -> list[Sentence]:
    if not words:
        return []
    turns: list[list[Word]] = [[words[0]]]
    for word in words[1:]:
        if word.speaker == turns[-1][-1].speaker:
            turns[-1].append(word)
        else:
            turns.append([word])
    sentences: list[Sentence] = []
    for turn in turns:
        sentences.extend(_split_turn(turn))
    return sentences


def canonical_speakers(sentences: list[Sentence]) -> list[Sentence]:
    labels = [sentence.speaker for sentence in sentences]
    if labels and all(_SPEAKER_LABEL.fullmatch(label) for label in labels):
        return sentences
    mapping: dict[str, str] = {}
    for sentence in sentences:
        if sentence.speaker not in mapping:
            mapping[sentence.speaker] = f"Speaker {len(mapping) + 1}"
        sentence.speaker = mapping[sentence.speaker]
    return sentences


def mark_overlaps(sentences: list[Sentence], min_overlap: float = MIN_OVERLAP_SEC) -> None:
    for sentence in sentences:
        sentence.overlap = False
    for index, left in enumerate(sentences):
        for right in sentences[index + 1 :]:
            if left.speaker == right.speaker:
                continue
            overlap = min(left.end, right.end) - max(left.start, right.start)
            if overlap >= min_overlap:
                left.overlap = True
                right.overlap = True


def _split_turn(words: list[Word]) -> list[Sentence]:
    text, spans = _join_words(words)
    pieces = _sentence_spans(text)
    if not pieces:
        return []
    results: list[Sentence] = []
    for char_start, char_end, sentence in pieces:
        covered = [
            (start, end, word)
            for start, end, word in spans
            if start < char_end and end > char_start
        ]
        if not covered:
            continue
        unique_words = {id(word) for _, _, word in covered}
        if len(unique_words) == 1 and len(pieces) > 1:
            word_start, word_end, word = covered[0]
            width = max(word_end - word_start, 1)
            duration = max(word.end - word.start, 0.0)
            start = word.start + duration * ((char_start - word_start) / width)
            end = word.start + duration * ((char_end - word_start) / width)
        else:
            start = covered[0][2].start
            end = covered[-1][2].end
        if end < start:
            end = start
        results.append(
            Sentence(speaker=words[0].speaker, start=start, end=end, text=sentence)
        )
    return results


def _join_words(words: list[Word]) -> tuple[str, list[tuple[int, int, Word]]]:
    parts: list[str] = []
    spans: list[tuple[int, int, Word]] = []
    cursor = 0
    for word in words:
        token = " ".join(word.text.split())
        if not token:
            continue
        if parts:
            cursor += 1
        start = cursor
        parts.append(token)
        cursor += len(token)
        spans.append((start, cursor, word))
    return " ".join(parts), spans


def _sentence_spans(text: str) -> list[tuple[int, int, str]]:
    normalized = " ".join(text.strip().split())
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for sentence in split_sentences(normalized):
        index = normalized.find(sentence, cursor)
        if index < 0:
            index = cursor
        end = index + len(sentence)
        spans.append((index, end, sentence))
        cursor = end
    return spans


def _is_abbrev_fragment(fragment: str) -> bool:
    match = re.search(r"([A-Za-z]+)\.$", fragment)
    if not match:
        return False
    word = match.group(1)
    if len(word) == 1 and word.isupper():
        return True
    return word.casefold() in _ABBREV
