"""Keep the word label, and flag a tone conflict when the voice disagrees."""

from __future__ import annotations

POLARITIES = ("positive", "negative")


def tone_polarity(valence: float | None, positive_at: float, negative_at: float) -> str:
    if valence is None:
        return "unknown"
    if valence > positive_at:
        return "positive"
    if valence < negative_at:
        return "negative"
    return "neutral"


def apply_tone(
    sentiment: str,
    text_confidence: float,
    tone_label: str,
    confidence_scale: float,
) -> tuple[bool, float]:
    conflict = (
        sentiment in POLARITIES
        and tone_label in POLARITIES
        and sentiment != tone_label
    )
    confidence = text_confidence * confidence_scale if conflict else text_confidence
    confidence = min(1.0, max(0.0, confidence))
    return conflict, round(confidence, 4)
