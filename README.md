# Video sentiment

Local pipeline for any English discussion video. It labels anonymous speakers, scores every subject they mention, and flags the line when the voice tone disagrees with the words.

The subjects are whatever people actually talk about. One sentence can praise one subject and criticize another at the same time. “They” stays in the output when the earlier turns do not make the referent clear. The stored sentiment always comes from the words. A positive line said in a negative tone stays positive and is marked `tone_conflict`.

## What you get

For each run, the output folder contains:

- `result.json` — speakers, timestamps, quotes, subjects, scores, and tone
- `utterances.csv` — one row per speaker, sentence, and subject
- `summary.csv` — counts and the dominant label per speaker and subject
- `audio.wav` — 16 kHz mono audio extracted from the video (`analyze` only)

A sentence with no subject still appears in `utterances.csv`, with the subject columns left blank.

## Setup

From `C:\Users\sanjjain\video-sentiment`:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev,ml]"
```

Install [Ollama](https://ollama.com) and pull the language model named in `config.yaml`:

```powershell
ollama pull llama3.1
```

Speaker labels need a Hugging Face read token. Accept the terms for these models, then set the token in the shell you use to run the tool:

- https://huggingface.co/pyannote/speaker-diarization-community-1
- https://huggingface.co/pyannote/segmentation-3.0

```powershell
$env:HF_TOKEN = "hf_your_token"
```

The first full run downloads the speech model (`large-v3`, about 3 GB), the speaker model, and the voice model. A GPU with 8 GB of memory is a comfortable fit. CPU works and is much slower. `config.yaml` can point `whisper_model` at `small` for a faster first pass.

`ffmpeg` does not have to be installed separately. The project uses a bundled copy when `ffmpeg` is not on PATH.

## Commands

Check what is installed:

```powershell
.\.venv\Scripts\video-sentiment check
```

Score the sample transcript. This needs Ollama and does not need a video, a GPU, or the speech models:

```powershell
.\.venv\Scripts\video-sentiment score-text examples\sample_turns.json -o outputs\sample
```

Run a video file:

```powershell
.\.venv\Scripts\video-sentiment analyze C:\path\to\discussion.mp4 -o outputs\discussion
```

Or a public YouTube link. The audio is downloaded into the output folder, then analyzed. Quote the link in PowerShell:

```powershell
.\.venv\Scripts\video-sentiment analyze "https://www.youtube.com/watch?v=VIDEO_ID" -o outputs\video1
```

Useful flags:

- `--min-speakers 3 --max-speakers 3` when you know how many people talk
- `--single-speaker` to skip speaker separation and call everyone Speaker 1
- `--skip-tone` to score the words only
- `--config path\to\config.yaml` to use a different settings file

`Speaker 1` is the first voice in that file. The same label in another file is not the same person.

## How a line is scored

1. The video becomes 16 kHz mono audio.
2. WhisperX writes the words, aligns them in time, and pyannote assigns `Speaker 1`, `Speaker 2`, and so on.
3. Each speaker turn is split into sentences.
4. A local voice model reads valence on that sentence. Above 0.65 is a positive tone, below 0.40 is a negative tone, and the middle is neutral.
5. A local Ollama model reads the sentence plus the previous eight sentences. It lists every subject, resolves “they” when it can, and returns `positive`, `negative`, `neutral`, or `mixed`.
6. If the words are positive and the tone is negative, or the reverse, `tone_conflict` is true and the confidence is multiplied by 0.75. The label stays the word label. Neutral or mixed wording does not become a conflict.

`summary.csv` rolls mentions up by speaker and subject. The same name in different letter case counts as one subject. If positive and negative mentions tie for the lead, the dominant label is `mixed`.

`name_hints` in `config.yaml` is an optional list of spellings the transcriber should listen for. It does not limit which subjects get a score. Leave it empty to cover any topic.

## Limits

Pronouns are the weak spot. Unresolved ones are kept as their own subject. Similar voices and people talking over each other can swap speaker numbers; overlapped lines are marked `overlap`. The tone flag catches a clear mismatch. Dry sarcasm in a flat voice often stays neutral and does not change the label.

The voice-model weights (`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`) are CC BY-NC-SA 4.0, so that stage is for non-commercial use. pyannote’s speaker model is CC BY 4.0.

## Tests

```powershell
.\.venv\Scripts\python -m pytest
```
