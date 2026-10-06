from pathlib import Path

from video_sentiment.download import default_output_dir, is_media_url, safe_folder_name, youtube_id
from video_sentiment.errors import VideoSentimentError


def test_youtube_links_are_urls():
    assert is_media_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert is_media_url("https://youtu.be/dQw4w9WgXcQ")
    assert is_media_url("https://www.youtube.com/shorts/abcdefghijk")


def test_local_paths_are_not_urls():
    assert not is_media_url(r"C:\videos\discussion.mp4")
    assert not is_media_url("videos/discussion.mp4")
    assert not is_media_url("")


def test_folder_name_drops_characters_windows_rejects():
    assert safe_folder_name('Boeing vs Airbus: 2026?') == "Boeing vs Airbus 2026"
    assert safe_folder_name("   ") == "video"


def test_youtube_id_from_common_links():
    assert youtube_id("https://youtu.be/DIeSOIRcyng") == "DIeSOIRcyng"
    assert youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert youtube_id("https://www.youtube.com/shorts/abcdefghijk") == "abcdefghijk"


def test_output_dir_uses_youtube_title(monkeypatch):
    monkeypatch.setattr(
        "video_sentiment.download.video_title",
        lambda url: "Boeing vs Airbus: In 2026?",
    )
    assert default_output_dir("https://youtu.be/DIeSOIRcyng") == Path("outputs") / "Boeing vs Airbus In 2026"


def test_output_dir_uses_video_id_when_title_is_missing(monkeypatch):
    def missing(url):
        raise VideoSentimentError("no title")

    monkeypatch.setattr("video_sentiment.download.video_title", missing)
    assert default_output_dir("https://youtu.be/DIeSOIRcyng") == Path("outputs") / "DIeSOIRcyng"


def test_output_dir_uses_local_file_name():
    assert default_output_dir(r"C:\videos\My Talk.mp4") == Path("outputs") / "My Talk"
