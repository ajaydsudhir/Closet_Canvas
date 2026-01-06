from services.capture.application.request_session_clip_upload import (
    _build_clip_object_key,
    _sanitize_clip_filename,
)


def test_sanitize_clip_filename_preserves_extension():
    clip_id = "a" * 32
    result = _sanitize_clip_filename("foo bar.webm", clip_id)
    assert result.endswith(".webm")
    assert clip_id in result
    assert "foo_bar" in result


def test_sanitize_clip_filename_adds_default_extension():
    clip_id = "b" * 32
    result = _sanitize_clip_filename("clip_without_ext", clip_id)
    assert result.endswith(".bin")


def test_build_clip_object_key_includes_prefix():
    clip_id = "c" * 32
    key = _build_clip_object_key("sessions/abc", "clip.mp4", clip_id)
    assert key.startswith("sessions/abc/clips/")
    assert key.endswith(".mp4")
