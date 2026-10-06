from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from video_sentiment.errors import VideoSentimentError
from video_sentiment.sentences import Sentence

SYSTEM_PROMPT = """You score aspect sentiment in an English conversation.
The words decide the sentiment.

For the CURRENT sentence, list every subject the speaker mentions or clearly refers to.
Subjects can be anything: companies, products, people, places, themes, or vague phrases such as "they", "their", "the industry", or "it".
Do not limit the list to any predefined names. Score whatever this sentence is about.
Use the previous turns only to decide what a vague phrase points to.

Sentiment is about that subject, from the words only:
- positive: praise, preference, advantage, recommendation
- negative: criticism, rejection, risk, disadvantage, or "not good"
- neutral: a fact, a question, or no evaluation
- mixed: the same subject is both praised and criticized in this sentence

A comparison produces one record per side. "A is safer than B" means A is positive and B is negative.
A sentence can contain several subjects with different sentiments.

When a vague phrase points to an earlier name, set resolution to "resolved" and resolved to that name.
When it does not, set resolution to "unresolved" and set resolved equal to the surface words.
A direct name is resolution "resolved".

Return JSON only, with this shape:
{"targets":[{"surface":"","resolved":"","resolution":"resolved","sentiment":"neutral","evidence":"short quote from the current sentence","text_confidence":0.0}]}
If the sentence mentions no subject, return {"targets":[]}.
"""

SENTIMENTS = {"positive", "negative", "neutral", "mixed"}
RESOLUTIONS = {"resolved", "unresolved"}
_VAGUE = {
    "they",
    "them",
    "their",
    "theirs",
    "it",
    "its",
    "he",
    "she",
    "this",
    "that",
    "those",
    "these",
}


def user_message(history: list[Sentence], current: Sentence) -> str:
    lines = ["Previous turns:"]
    if not history:
        lines.append("(none)")
    else:
        lines.extend(f"{item.speaker}: {item.text}" for item in history)
    lines.extend(["", f"Current sentence ({current.speaker}):", current.text])
    return "\n".join(lines)


def ollama_complete(url: str, model: str, system: str, user: str, timeout: float = 180) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": "json",
        "keep_alive": "30m",
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        _chat_url(url),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise VideoSentimentError(
            f"Ollama rejected the request ({exc.code}). {detail[:500]} "
            f"Pull the model with: ollama pull {model}"
        ) from exc
    except urllib.error.URLError as exc:
        raise VideoSentimentError(
            f"Ollama is not reachable at {url}. Start Ollama, then run: ollama pull {model}"
        ) from exc
    try:
        return str(payload["message"]["content"])
    except (KeyError, TypeError) as exc:
        raise VideoSentimentError("Ollama returned a response without message content.") from exc


def ollama_models(url: str, timeout: float = 10) -> list[str]:
    request = urllib.request.Request(_root(url) + "/api/tags", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise VideoSentimentError(f"Ollama is not reachable at {url}.") from exc
    models = payload.get("models", [])
    return [str(item.get("name", "")) for item in models if item.get("name")]


def parse_targets(raw: str) -> list[dict]:
    data = _load_json_object(raw)
    rows = data.get("targets", [])
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise VideoSentimentError("The model returned targets in an unexpected shape.")
    parsed: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        sentiment = str(row.get("sentiment", "")).strip().casefold()
        if sentiment not in SENTIMENTS:
            continue
        surface = " ".join(str(row.get("surface", "")).split())
        resolved = " ".join(str(row.get("resolved", "")).split())
        if not surface and not resolved:
            continue
        if not surface:
            surface = resolved
        if not resolved:
            resolved = surface
        resolution = str(row.get("resolution", "")).strip().casefold()
        if resolution not in RESOLUTIONS:
            vague = surface.casefold() in _VAGUE and resolved.casefold() == surface.casefold()
            resolution = "unresolved" if vague else "resolved"
        evidence = " ".join(str(row.get("evidence", "")).split())
        try:
            confidence = float(row.get("text_confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        parsed.append(
            {
                "surface": surface,
                "resolved": resolved,
                "resolution": resolution,
                "sentiment": sentiment,
                "evidence": evidence,
                "text_confidence": min(1.0, max(0.0, confidence)),
            }
        )
    return parsed


def _load_json_object(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise VideoSentimentError("The model did not return JSON.")
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise VideoSentimentError("The model did not return JSON.") from exc
    if not isinstance(data, dict):
        raise VideoSentimentError("The model JSON must be an object.")
    return data


def _root(url: str) -> str:
    return url.rstrip("/")


def _chat_url(url: str) -> str:
    return _root(url) + "/api/chat"
