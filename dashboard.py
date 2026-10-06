"""Browser view of finished video-sentiment runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

LABELS = (
    ("positive", "Positive", "#1f8a5b"),
    ("negative", "Negative", "#d0453f"),
    ("neutral", "Fact", "#6b7a8c"),
    ("mixed", "Mixed", "#d8962b"),
)
LABEL_WORD = {key: word for key, word, _ in LABELS}
LABEL_COLOR = {key: color for key, _, color in LABELS}
PLAIN = {
    "positive": "Mostly positive",
    "negative": "Mostly negative",
    "neutral": "Mostly facts",
    "mixed": "Mixed",
    "tie": "No clear winner",
}
HIDDEN_COLUMNS = {"source", "duration_sec", "speakers"}
TOP_SUBJECTS = 9


def main() -> None:
    st.set_page_config(page_title="Video sentiment", page_icon="▶", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='hero'>"
        "<h1>Video sentiment analysis</h1>"
        "<p>Choose a finished video and see every subject scored as positive, negative, fact, or mixed.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    _results_page()



def _playable(source: str) -> str | None:
    from video_sentiment.download import is_media_url

    if is_media_url(source):
        return source
    path = Path(source)
    return str(path) if path.is_file() else None



def _results_page() -> None:
    runs = _runs()
    if not runs:
        st.markdown(
            "<div class='empty'><p class='empty-title'>No finished runs yet</p>"
            "<p class='muted'>Finished videos show up here after an analysis.</p></div>",
            unsafe_allow_html=True,
        )
        return

    chosen = st.selectbox("Results", [path.name for path in runs])
    folder = next(path for path in runs if path.name == chosen)
    document = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    _ensure_result_csv(folder, document)
    utterances = document.get("utterances") or []
    summary = document.get("summary", {}).get("by_speaker") or {}

    source = str(document.get("source") or "")
    playable = _playable(source) if source else None
    left, right = st.columns([3, 2], gap="large")
    with left:
        if playable:
            st.video(playable)
        else:
            st.info("The original video is not available on this PC.")
    with right:
        st.markdown(f"<h3 class='side-title'>{_escape(folder.name)}</h3>", unsafe_allow_html=True)
        _headline(document, utterances, summary)

    _subject_cards(summary)
    _timeline(utterances, float(document.get("duration_sec") or 0))
    _results(folder)


def _runs() -> list[Path]:
    if not OUTPUTS.exists():
        return []
    found = [path for path in OUTPUTS.iterdir() if path.is_dir() and (path / "result.json").exists()]
    return sorted(found, key=lambda path: (path / "result.json").stat().st_mtime, reverse=True)


def _ensure_result_csv(folder: Path, document: dict) -> None:
    target = folder / "result.csv"
    if target.exists() and target.stat().st_mtime >= (folder / "result.json").stat().st_mtime:
        return
    from video_sentiment.export import RESULT_FIELDS, _write_csv, result_rows

    _write_csv(target, RESULT_FIELDS, result_rows(document))


def _headline(document: dict, utterances: list[dict], summary: dict) -> None:
    rows = _flat(utterances)
    subjects = {name for targets in summary.values() for name in targets}
    speakers = document.get("speakers") or []
    counts = {key: sum(1 for row in rows if row["sentiment"] == key) for key, _, _ in LABELS}
    tiles = [
        ("Length", _clock(float(document.get("duration_sec") or 0))),
        ("Sentences", str(len(utterances))),
        ("Subjects", str(len(subjects))),
        ("Speakers", str(len(speakers) or 1)),
    ]
    st.markdown(
        "<div class='tiles'>"
        + "".join(f"<div class='tile'><p>{label}</p><strong>{value}</strong></div>" for label, value in tiles)
        + "</div>",
        unsafe_allow_html=True,
    )
    total = sum(counts.values()) or 1
    bar = "".join(
        f'<span style="width:{counts[key] / total * 100:.1f}%;background:{color}"></span>'
        for key, _, color in LABELS
        if counts[key]
    )
    legend = "".join(
        f"<span class='legend'><i style='background:{color}'></i>{word} {counts[key]}</span>"
        for key, word, color in LABELS
    )
    st.markdown(
        f"<p class='muted tight'>Every mention in this video</p><div class='bar big'>{bar}</div>"
        f"<div class='legends'>{legend}</div>",
        unsafe_allow_html=True,
    )


def _subject_cards(summary: dict) -> None:
    cards = []
    for speaker, targets in summary.items():
        for name, counts in targets.items():
            mentions = sum(int(counts.get(key, 0)) for key, _, _ in LABELS)
            cards.append((mentions, speaker, name, counts))
    cards.sort(key=lambda card: (-card[0], card[2].casefold()))
    st.markdown("<h2>Subjects</h2>", unsafe_allow_html=True)
    if not cards:
        st.caption("No subject was named.")
        return
    st.markdown(
        "<p class='muted'>The subjects mentioned most come first.</p>",
        unsafe_allow_html=True,
    )
    _card_grid(cards[:TOP_SUBJECTS])
    rest = cards[TOP_SUBJECTS:]
    if rest:
        with st.expander(f"Show the other {len(rest)} subjects"):
            _card_grid(rest)


def _card_grid(cards: list[tuple]) -> None:
    columns = st.columns(3, gap="medium")
    for index, (_, speaker, name, counts) in enumerate(cards):
        with columns[index % 3]:
            st.markdown(_card(speaker, name, counts), unsafe_allow_html=True)


def _card(speaker: str, name: str, counts: dict) -> str:
    total = sum(int(counts.get(key, 0)) for key, _, _ in LABELS) or 1
    pieces = []
    bits = []
    for key, word, color in LABELS:
        count = int(counts.get(key, 0))
        if count:
            pieces.append(f'<span style="width:{(count / total) * 100:.1f}%;background:{color}"></span>')
            bits.append(f"{count} {word.lower()}")
    winner = PLAIN.get(str(counts.get("dominant")), str(counts.get("dominant") or ""))
    color = LABEL_COLOR.get(str(counts.get("dominant")), "#6b7a8c")
    return (
        f"<div class='card' style='border-top:5px solid {color}'>"
        f"<p class='who'>{_escape(speaker)}</p>"
        f"<h3>{_escape(name)}</h3>"
        f"<span class='pill' style='background:{color}'>{_escape(winner)}</span>"
        f"<div class='bar'>{''.join(pieces)}</div>"
        f"<p class='counts'>{_escape(' · '.join(bits))}</p>"
        "</div>"
    )


def _timeline(utterances: list[dict], duration: float) -> None:
    st.markdown("<h2>Along the video</h2>", unsafe_allow_html=True)
    if not utterances:
        st.caption("No sentences were heard.")
        return
    span = duration or max(float(row["end"]) for row in utterances)
    ticks = []
    for row in utterances:
        start = float(row["start"])
        width = max((float(row["end"]) - start) / span * 100, 0.8)
        left = start / span * 100
        color = LABEL_COLOR.get(_row_sentiment(row), "#6b7a8c")
        ticks.append(
            f'<span class="tick" style="left:{left:.2f}%;width:{width:.2f}%;background:{color}" '
            f'title="{_clock(start)}  {_escape(row["text"])}"></span>'
        )
    st.markdown(
        f"<div class='track'>{''.join(ticks)}</div>"
        f"<div class='scale'><span>0:00</span><span>{_clock(span / 2)}</span><span>{_clock(span)}</span></div>",
        unsafe_allow_html=True,
    )
    options = [f"{_clock(float(row['start']))}  {row['text']}" for row in utterances]
    pick = st.selectbox("Read a sentence", options)
    row = utterances[options.index(pick)]
    chips = []
    for target in row.get("targets") or []:
        sentiment = str(target.get("sentiment") or "")
        chips.append(
            f"<span class='chip' style='background:{LABEL_COLOR.get(sentiment, '#6b7a8c')}'>"
            f"{_escape(target.get('resolved') or target.get('surface') or 'subject')} · "
            f"{LABEL_WORD.get(sentiment, sentiment)}"
            "</span>"
        )
    evidence = next((str(target.get("evidence") or "") for target in row.get("targets") or [] if target.get("evidence")), "")
    st.markdown(
        "<div class='quote'>"
        f"<p class='when'>{_escape(row['speaker'])} · {_clock(float(row['start']))}–{_clock(float(row['end']))}</p>"
        f"<p class='said'>{_escape(row['text'])}</p>"
        f"<p class='chips'>{''.join(chips) or '<span class=\"chip quiet\">No subject</span>'}</p>"
        f"<p class='evidence'>{_escape(evidence)}</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def _results(folder: Path) -> None:
    st.markdown("<h2>Results</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p class='muted'>result.csv has one row per sentence and subject. summary.csv has one row per subject.</p>",
        unsafe_allow_html=True,
    )
    order = ["result.csv", "summary.csv", "utterances.csv", "result.json"]
    present = [name for name in order if (folder / name).exists()]
    tabs = st.tabs(present)
    for tab, name in zip(tabs, present):
        path = folder / name
        with tab:
            st.download_button(f"Download {name}", path.read_bytes(), file_name=name, key=f"dl-{folder.name}-{name}")
            if name.endswith(".csv"):
                frame = pd.read_csv(path, encoding="utf-8-sig")
                frame = frame.drop(columns=[column for column in frame.columns if column in HIDDEN_COLUMNS])
                st.dataframe(frame, width="stretch", hide_index=True, height=420)
            else:
                st.json(json.loads(path.read_text(encoding="utf-8")), expanded=1)


def _flat(utterances: list[dict]) -> list[dict]:
    rows = []
    for utterance in utterances:
        for target in utterance.get("targets") or []:
            rows.append({"sentiment": target.get("sentiment") or ""})
    return rows


def _row_sentiment(row: dict) -> str:
    targets = row.get("targets") or []
    if not targets:
        return "neutral"
    labels = {str(target.get("sentiment") or "neutral") for target in targets}
    if {"positive", "negative"} <= labels or "mixed" in labels:
        return "mixed"
    for label in ("negative", "positive"):
        if label in labels:
            return label
    return "neutral"


def _clock(seconds: float) -> str:
    whole = max(int(round(seconds)), 0)
    hours, rest = divmod(whole, 3600)
    if hours:
        return f"{hours}:{rest // 60:02d}:{rest % 60:02d}"
    return f"{rest // 60}:{rest % 60:02d}"


def _escape(value: object) -> str:
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


_CSS = """
<style>
    .block-container { padding-top: 1.6rem; max-width: 1240px; }
    h2 { margin-top: 2.2rem; font-size: 1.6rem; letter-spacing: -0.02em; }
    h3 { margin: 0.2rem 0 0.6rem; font-size: 1.3rem; line-height: 1.25; }
    .hero {
        background: linear-gradient(130deg, #10352a 0%, #1f7a4d 55%, #d8962b 120%);
        color: #fdf8ef; border-radius: 26px; padding: 2.1rem 2.4rem 1.9rem; margin-bottom: 1.2rem;
        box-shadow: 0 22px 50px rgba(16, 53, 42, 0.25);
    }
    .hero h1 { color: #fffaf1; font-size: 2.5rem; letter-spacing: -0.035em; margin: 0.15rem 0 0.4rem; line-height: 1.1; }
    .hero p { color: rgba(253, 248, 239, 0.86); font-size: 1.08rem; margin: 0; }
    .hero .kicker { letter-spacing: 0.18em; text-transform: uppercase; font-weight: 800; font-size: 0.75rem; color: #f6d48f; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1.05rem; font-weight: 700; padding: 0.6rem 1.1rem; border-radius: 12px 12px 0 0; }
    .side-title { font-size: 1.45rem; margin-top: 0; }
    .muted { color: #5f6b78; margin: 0 0 0.6rem; }
    .muted.tight { margin: 1rem 0 0.3rem; font-size: 0.9rem; }
    .empty { background: #fffdf8; border: 2px dashed #d9cdb8; border-radius: 22px; padding: 2.4rem; text-align: center; margin-top: 0.8rem; }
    .empty-title { font-size: 1.35rem; font-weight: 800; margin: 0 0 0.3rem; }
    .tiles { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.7rem; }
    .tile { background: #fffdf8; border-radius: 18px; padding: 0.85rem 1rem; box-shadow: 0 8px 24px rgba(60, 42, 20, 0.07); }
    .tile p { margin: 0; color: #6b7a8c; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }
    .tile strong { font-size: 1.9rem; letter-spacing: -0.03em; color: #10352a; }
    .legends { display: flex; flex-wrap: wrap; gap: 0.9rem; font-size: 0.88rem; color: #3b4652; }
    .legend i { display: inline-block; width: 11px; height: 11px; border-radius: 4px; margin-right: 0.35rem; vertical-align: -1px; }
    .card {
        background: #fffdf8; border-radius: 20px; padding: 1rem 1.1rem 0.95rem; margin-bottom: 1rem;
        box-shadow: 0 10px 28px rgba(60, 42, 20, 0.07); min-height: 172px; transition: transform .15s ease, box-shadow .15s ease;
    }
    .card:hover { transform: translateY(-3px); box-shadow: 0 16px 36px rgba(60, 42, 20, 0.13); }
    .who, .counts { color: #6b7a8c; margin: 0; font-size: 0.88rem; }
    .pill { display: inline-block; color: white; border-radius: 999px; padding: 0.2rem 0.7rem; font-size: 0.8rem; font-weight: 800; }
    .bar { display: flex; height: 12px; border-radius: 999px; overflow: hidden; background: #ece5d8; margin: 0.8rem 0 0.45rem; }
    .bar.big { height: 18px; margin: 0.2rem 0 0.6rem; }
    .bar span { display: block; height: 100%; }
    .track { position: relative; height: 34px; border-radius: 999px; background: linear-gradient(90deg, #e9e1d3, #f1ebe0); margin: 0.6rem 0 0.35rem; }
    .tick { position: absolute; top: 7px; height: 20px; border-radius: 6px; min-width: 6px; opacity: 0.92; cursor: help; }
    .tick:hover { opacity: 1; transform: scaleY(1.25); }
    .scale { display: flex; justify-content: space-between; color: #6b7a8c; font-size: 0.82rem; margin-bottom: 0.8rem; }
    .quote { background: #fffdf8; border-radius: 20px; padding: 1.05rem 1.2rem 1rem; box-shadow: 0 10px 28px rgba(60, 42, 20, 0.07); }
    .when, .evidence { color: #6b7a8c; margin: 0; }
    .said { font-size: 1.28rem; line-height: 1.4; margin: 0.4rem 0 0.7rem; }
    .chips { margin: 0; }
    .chip { display: inline-block; color: white; border-radius: 999px; padding: 0.28rem 0.75rem; font-size: 0.84rem; font-weight: 800; margin: 0.2rem 0.4rem 0.2rem 0; }
    .chip.quiet { background: #8d97a3; }
    .evidence { margin-top: 0.55rem; }
    .job-name { font-weight: 800; font-size: 1.05rem; margin: 0.6rem 0 0.2rem; }
    [data-testid="stVideo"] { border-radius: 20px; overflow: hidden; box-shadow: 0 16px 40px rgba(16, 53, 42, 0.18); }
    [data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
</style>
"""


if __name__ == "__main__":
    main()
