import json

import pytest

from video_sentiment.errors import VideoSentimentError
from video_sentiment.sentiment import parse_targets, user_message
from video_sentiment.sentences import Sentence


def test_user_message_includes_history_and_current_speaker():
    history = [Sentence("Speaker 1", 0, 1, "Airbus is the European one.")]
    current = Sentence("Speaker 2", 1, 2, "They are not good.")
    text = user_message(history, current)
    assert "Speaker 1: Airbus is the European one." in text
    assert "Current sentence (Speaker 2):" in text
    assert "They are not good." in text


def test_parse_targets_clamps_and_fills_missing_resolution():
    raw = """
    ```json
    {"targets":[
      {"surface":"Airbus","resolved":"Airbus","sentiment":"positive","evidence":"best cabin","text_confidence":1.4},
      {"surface":"they","resolved":"","resolution":"unresolved","sentiment":"negative","evidence":"not good","text_confidence":0.4},
      {"surface":"noise","resolved":"noise","sentiment":"happy","evidence":"nope","text_confidence":0.2}
    ]}
    ```
    """
    targets = parse_targets(raw)
    assert len(targets) == 2
    assert targets[0]["resolution"] == "resolved"
    assert targets[0]["text_confidence"] == 1.0
    assert targets[1]["resolved"] == "they"
    assert targets[1]["resolution"] == "unresolved"


def test_parse_targets_rejects_non_json():
    with pytest.raises(VideoSentimentError):
        parse_targets("no json here")


def test_empty_subject_list():
    assert parse_targets(json.dumps({"targets": []})) == []
