from video_sentiment.download import is_media_url


def test_youtube_links_are_urls():
    assert is_media_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert is_media_url("https://youtu.be/dQw4w9WgXcQ")
    assert is_media_url("https://www.youtube.com/shorts/abcdefghijk")


def test_local_paths_are_not_urls():
    assert not is_media_url(r"C:\videos\discussion.mp4")
    assert not is_media_url("videos/discussion.mp4")
    assert not is_media_url("")
