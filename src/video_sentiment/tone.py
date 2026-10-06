from __future__ import annotations

from pathlib import Path

from video_sentiment.audio import read_wav
from video_sentiment.config import Settings
from video_sentiment.errors import VideoSentimentError
from video_sentiment.fuse import tone_polarity
from video_sentiment.sentences import Sentence


def tag_sentences(wav: Path, sentences: list[Sentence], settings: Settings) -> None:
    if not sentences:
        return
    model, processor, torch, device = _load(settings)
    samples, rate = read_wav(wav)
    if rate != 16000:
        raise VideoSentimentError(f"{wav} is {rate} Hz. The tone model expects 16000 Hz.")
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    for sentence in sentences:
        start = max(0, int(sentence.start * rate))
        end = min(len(audio), int(sentence.end * rate))
        if end - start < int(0.25 * rate):
            sentence.tone_valence = None
            sentence.tone_polarity = "unknown"
            continue
        clip = audio[start:end]
        values = processor(clip, sampling_rate=rate)
        waveform = values["input_values"][0]
        tensor = torch.from_numpy(waveform).to(device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            _hidden, logits = model(tensor)
        vector = logits.detach().float().cpu().numpy().reshape(-1)
        if vector.size < 3:
            raise VideoSentimentError("The tone model returned fewer than 3 scores.")
        # Model order is arousal, dominance, valence.
        valence = float(vector[2])
        sentence.tone_valence = valence
        sentence.tone_polarity = tone_polarity(
            valence,
            settings.valence_positive,
            settings.valence_negative,
        )


def _load(settings: Settings):
    from video_sentiment.compat import trust_local_model_checkpoints

    trust_local_model_checkpoints()
    try:
        import torch
        from transformers import Wav2Vec2Processor
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )
    except ImportError as exc:
        raise VideoSentimentError(
            'The tone model dependencies are not installed. Run: pip install -e ".[ml]"'
        ) from exc

    import torch.nn as nn

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(getattr(config, "final_dropout", 0.0))
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features):
            hidden = self.dropout(features)
            hidden = torch.tanh(self.dense(hidden))
            hidden = self.dropout(hidden)
            return self.out_proj(hidden)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            pooled = outputs[0].mean(dim=1)
            return pooled, self.classifier(pooled)

    device = settings.resolved_device()
    processor = Wav2Vec2Processor.from_pretrained(settings.tone_model)
    model = EmotionModel.from_pretrained(settings.tone_model).to(device)
    model.eval()
    return model, processor, torch, device
