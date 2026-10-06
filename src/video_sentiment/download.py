from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from video_sentiment.errors import VideoSentimentError

_INVALID_FOLDER_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_FOLDER_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{n}" for n in range(1, 10)),
    *(f"LPT{n}" for n in range(1, 10)),
}


def is_media_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def safe_folder_name(name: str) -> str:
    cleaned = _INVALID_FOLDER_CHARS.sub(" ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    cleaned = cleaned[:80].rstrip(" .")
    if not cleaned or cleaned.upper() in _RESERVED_FOLDER_NAMES:
        return "video"
    return cleaned


def youtube_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().removeprefix("www.")
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
        return video_id or None
    if host in {"youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com"}:
        query_id = parse_qs(parsed.query).get("v", [""])[0]
        if query_id:
            return query_id
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
            return parts[1]
    return None


def video_title(url: str) -> str:
    try:
        import yt_dlp
    except ImportError as exc:
        raise VideoSentimentError(
            'The link downloader is not installed. From the project folder run: pip install yt-dlp'
        ) from exc

    from yt_dlp.utils import DownloadError

    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        raise VideoSentimentError(f"Could not read the video title. Details: {exc}") from exc
    if not info:
        raise VideoSentimentError("Could not read the video title.")
    if info.get("entries"):
        info = next((item for item in info["entries"] if item), None)
        if not info:
            raise VideoSentimentError("Could not read the video title.")
    title = str(info.get("title") or "").strip()
    if not title:
        raise VideoSentimentError("Could not read the video title.")
    return title


def default_output_dir(source: str) -> Path:
    source = source.strip()
    if is_media_url(source):
        label = ""
        try:
            label = video_title(source)
        except VideoSentimentError:
            label = ""
        name = safe_folder_name(label)
        if name == "video":
            name = safe_folder_name(youtube_id(source) or "video")
        return Path("outputs") / name
    stem = Path(source).stem or Path(source).name or "video"
    return Path("outputs") / safe_folder_name(stem)


def download_audio(url: str, out_dir: Path) -> Path:
    try:
        import yt_dlp
    except ImportError as exc:
        raise VideoSentimentError(
            'The link downloader is not installed. From the project folder run: pip install yt-dlp'
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True,
        "restrictfilenames": True,
        "quiet": False,
    }
    from yt_dlp.utils import DownloadError

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            path = _downloaded_path(ydl, info, out_dir)
    except DownloadError as exc:
        raise VideoSentimentError(
            "Could not download that link. Use a public video URL, "
            "or save the file and pass its path. "
            f"Details: {exc}"
        ) from exc
    print(f"Downloaded {path}", flush=True)
    return path


def _downloaded_path(ydl, info, out_dir: Path) -> Path:
    if not info:
        raise VideoSentimentError("The link did not return a video.")
    if info.get("entries"):
        info = next((item for item in info["entries"] if item), None)
        if not info:
            raise VideoSentimentError("The link did not return a video.")
    path = Path(ydl.prepare_filename(info))
    if path.exists():
        return path
    matches = sorted(out_dir.glob("source.*"))
    if matches:
        return matches[0]
    raise VideoSentimentError("The download finished without a saved audio file.")
