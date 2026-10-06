"""Load official speech checkpoints on PyTorch 2.6 and newer."""

from __future__ import annotations


def trust_local_model_checkpoints() -> None:
    import torch

    if getattr(torch.load, "_video_sentiment_patched", False):
        return
    original = torch.load

    def load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    load._video_sentiment_patched = True  # type: ignore[attr-defined]
    torch.load = load  # type: ignore[method-assign]


def accept_legacy_hub_token() -> None:
    """Let older speaker-model code pass use_auth_token into current Hugging Face downloads."""
    import huggingface_hub
    from huggingface_hub import file_download

    modules = [huggingface_hub, file_download]
    try:
        import pyannote.audio.core.model as model_module
        import pyannote.audio.core.pipeline as pipeline_module
    except ImportError:
        model_module = None
        pipeline_module = None
    else:
        modules.extend([model_module, pipeline_module])

    originals = []
    for module in modules:
        current = getattr(module, "hf_hub_download", None)
        if current is None or getattr(current, "_video_sentiment_token", False):
            continue
        originals.append((module, current))

    wrapped: dict[int, object] = {}
    for module, current in originals:
        key = id(current)
        if key not in wrapped:
            wrapped[key] = _with_token_alias(current)
        setattr(module, "hf_hub_download", wrapped[key])


def _with_token_alias(download):
    def inner(*args, **kwargs):
        if "use_auth_token" in kwargs:
            token = kwargs.pop("use_auth_token")
            if token is not None and "token" not in kwargs:
                kwargs["token"] = token
        return download(*args, **kwargs)

    inner._video_sentiment_token = True  # type: ignore[attr-defined]
    return inner
