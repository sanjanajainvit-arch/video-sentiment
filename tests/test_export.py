import csv
import json

from video_sentiment.export import build_document, dominant, write_outputs
from video_sentiment.sentences import Sentence, Target


def _target(resolved: str, sentiment: str, confidence: float = 0.8) -> Target:
    return Target(
        surface=resolved,
        resolved=resolved,
        resolution="resolved",
        sentiment=sentiment,
        evidence=resolved,
        text_confidence=confidence,
        tone_polarity="neutral",
        tone_valence=0.5,
        tone_conflict=False,
        sentiment_confidence=confidence,
    )


def test_dominant_labels():
    assert dominant({"positive": 3, "negative": 1, "neutral": 0, "mixed": 0}) == "positive"
    assert dominant({"positive": 2, "negative": 2, "neutral": 0, "mixed": 0}) == "mixed"
    assert dominant({"positive": 1, "negative": 0, "neutral": 1, "mixed": 0}) == "tie"
    assert dominant({"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}) == "neutral"


def test_summary_merges_case_and_csv_keeps_subjectless_lines(tmp_path):
    sentences = [
        Sentence(
            "Speaker 1",
            0.0,
            2.0,
            "Airbus is wonderful.",
            tone_valence=0.2,
            tone_polarity="negative",
            targets=[_target("Airbus", "positive", 0.8)],
        ),
        Sentence(
            "Speaker 1",
            2.0,
            4.0,
            "airbus is still my pick.",
            targets=[_target("airbus", "positive", 0.7)],
        ),
        Sentence("Speaker 2", 4.0, 5.0, "Hello everyone."),
    ]
    sentences[0].targets[0].tone_conflict = True
    sentences[0].targets[0].sentiment_confidence = 0.6
    document = build_document("discussion.mp4", 5.0, sentences)
    airbus = document["summary"]["by_speaker"]["Speaker 1"]["Airbus"]
    assert airbus["positive"] == 2
    assert airbus["dominant"] == "positive"
    assert document["utterances"][0]["targets"][0]["tone_conflict"] is True

    write_outputs(document, tmp_path)
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["speakers"] == ["Speaker 1", "Speaker 2"]
    with (tmp_path / "utterances.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert rows[0]["sentiment"] == "positive"
    assert rows[0]["tone_conflict"] == "True"
    assert rows[2]["speaker"] == "Speaker 2"
    assert rows[2]["resolved"] == ""
    with (tmp_path / "summary.csv").open(encoding="utf-8-sig", newline="") as handle:
        summary = list(csv.DictReader(handle))
    assert summary[0]["target"] == "Airbus"
    assert summary[0]["mentions"] == "2"
    assert summary[0]["dominant"] == "positive"
