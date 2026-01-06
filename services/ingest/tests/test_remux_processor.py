from __future__ import annotations

from pathlib import Path

import pytest

from services.ingest.infrastructure import remux


def test_ffmpeg_processor_skips_non_live(tmp_path, monkeypatch):
    called = {"run": False}

    def fake_run(*args, **kwargs):
        called["run"] = True
        raise AssertionError("should not be called")

    monkeypatch.setattr(remux.subprocess, "run", fake_run)
    processor = remux.FFmpegClipProcessor(live_source_only=True)
    workdir = tmp_path / "work"
    workdir.mkdir()
    source = workdir / "input.bin"
    source.write_bytes(b"data")

    result = processor.process(
        source=source, workdir=workdir, metadata={"source": "manual"}
    )

    assert result == source
    assert called["run"] is False


def test_ffmpeg_processor_runs_for_live(tmp_path, monkeypatch):
    recorded = {}

    def fake_run(cmd, capture_output):
        recorded["cmd"] = cmd

        class _Result:
            returncode = 0
            stderr = b""

        # simulate ffmpeg writing the file
        Path(cmd[-1]).write_bytes(b"remuxed")
        return _Result()

    monkeypatch.setattr(remux.subprocess, "run", fake_run)
    processor = remux.FFmpegClipProcessor(target_extension=".mp4")
    workdir = tmp_path / "work"
    workdir.mkdir()
    source = workdir / "input.bin"
    source.write_bytes(b"data")

    result = processor.process(
        source=source, workdir=workdir, metadata={"source": "live"}
    )

    assert result.exists()
    assert result.suffix == ".mp4"
    assert recorded["cmd"][0] == "ffmpeg"
    assert recorded["cmd"][-1] == result.as_posix()


def test_ffmpeg_processor_raises_on_failure(tmp_path, monkeypatch):
    class _Result:
        returncode = 1
        stderr = b"boom"

    monkeypatch.setattr(remux.subprocess, "run", lambda *args, **kwargs: _Result())
    processor = remux.FFmpegClipProcessor()
    workdir = tmp_path / "work"
    workdir.mkdir()
    source = workdir / "input.bin"
    source.write_bytes(b"data")

    with pytest.raises(remux.FFmpegRemuxError):
        processor.process(source=source, workdir=workdir, metadata={"source": "live"})
