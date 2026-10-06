import pytest

from video_sentiment.sentences import (
    Word,
    canonical_speakers,
    mark_overlaps,
    split_sentences,
    words_to_sentences,
)
from video_sentiment.sentences import Sentence


def test_abbreviation_stays_in_the_sentence():
    assert split_sentences("Dr. Smith says Airbus is strong. Boeing is late.") == [
        "Dr. Smith says Airbus is strong.",
        "Boeing is late.",
    ]


def test_no_is_its_own_sentence():
    assert split_sentences("No. I prefer Airbus.") == ["No.", "I prefer Airbus."]


def test_speaker_labels_follow_first_appearance():
    words = [
        Word("Boeing", 0.0, 0.4, "SPEAKER_01"),
        Word("is", 0.4, 0.6, "SPEAKER_01"),
        Word("late.", 0.6, 1.0, "SPEAKER_01"),
        Word("Airbus", 1.2, 1.6, "SPEAKER_00"),
        Word("is", 1.6, 1.8, "SPEAKER_00"),
        Word("great.", 1.8, 2.2, "SPEAKER_00"),
    ]
    sentences = canonical_speakers(words_to_sentences(words))
    assert [(item.speaker, item.text) for item in sentences] == [
        ("Speaker 1", "Boeing is late."),
        ("Speaker 2", "Airbus is great."),
    ]
    assert sentences[0].start == pytest.approx(0.0)
    assert sentences[0].end == pytest.approx(1.0)
    assert sentences[1].start == pytest.approx(1.2)


def test_existing_speaker_labels_are_kept():
    sentences = canonical_speakers(
        [
            Sentence("Speaker 2", 0.0, 1.0, "Hello."),
            Sentence("Speaker 1", 1.0, 2.0, "Airbus is fine."),
        ]
    )
    assert [item.speaker for item in sentences] == ["Speaker 2", "Speaker 1"]


def test_one_timestamp_span_is_split_across_sentences():
    words = [Word("Airbus is great. Boeing is late.", 0.0, 10.0, "SPEAKER_00")]
    sentences = words_to_sentences(words)
    assert [item.text for item in sentences] == ["Airbus is great.", "Boeing is late."]
    assert sentences[0].start == pytest.approx(0.0)
    assert sentences[0].end < sentences[1].start
    assert sentences[1].end == pytest.approx(10.0)


def test_overlap_requires_different_speakers():
    sentences = [
        Sentence("Speaker 1", 0.0, 5.0, "Airbus is strong."),
        Sentence("Speaker 2", 4.6, 8.0, "I disagree."),
        Sentence("Speaker 1", 8.2, 10.0, "Still true."),
    ]
    mark_overlaps(sentences)
    assert sentences[0].overlap is True
    assert sentences[1].overlap is True
    assert sentences[2].overlap is False
