from video_sentiment.fuse import apply_tone, tone_polarity


def test_valence_bands():
    assert tone_polarity(0.8, 0.65, 0.40) == "positive"
    assert tone_polarity(0.2, 0.65, 0.40) == "negative"
    assert tone_polarity(0.5, 0.65, 0.40) == "neutral"
    assert tone_polarity(None, 0.65, 0.40) == "unknown"


def test_opposite_tone_flags_conflict_and_scales_confidence():
    conflict, confidence = apply_tone("positive", 0.8, "negative", 0.75)
    assert conflict is True
    assert confidence == 0.6


def test_neutral_words_keep_the_label_when_the_voice_is_angry():
    conflict, confidence = apply_tone("neutral", 0.7, "negative", 0.75)
    assert conflict is False
    assert confidence == 0.7


def test_mixed_words_are_not_a_tone_conflict():
    conflict, confidence = apply_tone("mixed", 0.4, "positive", 0.75)
    assert conflict is False
    assert confidence == 0.4
