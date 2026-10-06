from pathlib import Path

from video_sentiment.config import load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_repo_config_loads(monkeypatch):
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OLLAMA_MODEL", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("video_sentiment.config.user_environment", lambda name: None)
    settings = load_settings(ROOT / "config.yaml")
    assert settings.whisper_model == "large-v3"
    assert settings.language == "en"
    assert settings.context_sentences == 8
    assert settings.valence_positive == 0.65
    assert settings.valence_negative == 0.40
    assert settings.conflict_confidence_scale == 0.75
    assert settings.ollama_model == "llama3.1"
    assert settings.name_hints == []
    assert "Airbus" not in settings.initial_prompt
    assert "Boeing" not in settings.initial_prompt
    assert settings.hf_token is None


def test_saved_user_token_is_used(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(
        "video_sentiment.config.user_environment",
        lambda name: "hf_saved" if name == "HF_TOKEN" else None,
    )
    settings = load_settings(ROOT / "config.yaml")
    assert settings.hf_token == "hf_saved"


def test_name_hints_are_optional_spelling_help():
    from video_sentiment.config import Settings

    settings = Settings(name_hints=["Contoso", "Northwind"])
    prompt = settings.asr_prompt()
    assert prompt is not None
    assert "Contoso" in prompt
    assert "Northwind" in prompt


def test_environment_overrides_config(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11435")
    settings = load_settings(ROOT / "config.yaml")
    assert settings.hf_token == "hf_test"
    assert settings.ollama_model == "qwen2.5"
    assert settings.ollama_url == "http://127.0.0.1:11435"
