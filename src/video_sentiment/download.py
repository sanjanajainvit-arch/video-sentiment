from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from video_sentiment.errors import VideoSentimentError


def is_media_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


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
