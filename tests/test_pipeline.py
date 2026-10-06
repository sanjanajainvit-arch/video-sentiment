import json

import pytest

from video_sentiment.config import Settings
from video_sentiment.pipeline import load_turns, score_sentences
from video_sentiment.sentences import Sentence


def test_score_keeps_word_label_and_flags_tone(tmp_path):
    sentence = Sentence(
        "Speaker 1",
        0.0,
        2.0,
        "Airbus is wonderful.",
        tone_polarity="negative",
        tone_valence=0.1,
    )

    def complete(system, user):
        assert "Airbus is wonderful." in user
        return json.dumps(
            {
                "targets": [
                    {
                        "surface": "Airbus",
                        "resolved": "Airbus",
                        "resolution": "resolved",
                        "sentiment": "positive",
                        "evidence": "wonderful",
                        "text_confidence": 0.8,
                    }
                ]
            }
        )

    score_sentences([sentence], Settings(), complete=complete)
    target = sentence.targets[0]
    assert target.sentiment == "positive"
    assert target.tone_conflict is True
    assert target.sentiment_confidence == pytest.approx(0.6)


def test_invalid_json_is_retried():
    calls = {"n": 0}

    def complete(system, user):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json"
        return json.dumps({"targets": []})

    sentence = Sentence("Speaker 1", 0.0, 1.0, "Hello.")
    score_sentences([sentence], Settings(), complete=complete)
    assert calls["n"] == 2
    assert sentence.targets == []


def test_load_sample_turns():
    sentences = load_turns(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "examples" / "sample_turns.json"
    )
    assert [item.speaker for item in sentences] == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 2",
        "Speaker 1",
        "Speaker 3",
    ]
    assert sentences[1].text == "I would not go with them."
    assert "not good" in sentences[2].text


def test_missing_transcript(tmp_path):
    from video_sentiment.errors import VideoSentimentError

    with pytest.raises(VideoSentimentError):
        load_turns(tmp_path / "missing.json")
