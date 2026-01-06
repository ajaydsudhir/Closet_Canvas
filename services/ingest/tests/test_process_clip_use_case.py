from pathlib import Path

from services.ingest.application.use_cases.process_clip import ProcessClipUseCase
from services.ingest.domain.clip import ClipJob


class FakeStorage:
    def __init__(self, payloads: list[bytes] | None = None) -> None:
        self.uploaded: dict[str, object] = {}
        self._payloads = payloads or [b"fake-video-bytes"]

    def download(self, bucket: str, object_key: str, destination_path: str) -> None:
        data = self._payloads.pop(0)
        Path(destination_path).write_bytes(data)

    def upload(self, bucket: str, object_key: str, source_path: str) -> None:
        self.uploaded["bucket"] = bucket
        self.uploaded["key"] = object_key
        self.uploaded["content"] = Path(source_path).read_bytes()


class NoOpProcessor:
    def process(self, *, source: Path, workdir: Path, metadata=None):
        return source


class FakeHeaderCache:
    def __init__(self) -> None:
        self._storage: dict[str, bytes] = {}

    def set(self, session_id: str, header: bytes) -> None:
        self._storage[session_id] = header

    def get(self, session_id: str) -> bytes | None:
        return self._storage.get(session_id)


def test_process_clip_use_case_roundtrip():
    storage = FakeStorage()
    use_case = ProcessClipUseCase(storage=storage, processor=NoOpProcessor())
    job = ClipJob(
        session_id="session",
        clip_id="clip123",
        source_bucket="input",
        object_key="clip.mp4",
        target_bucket="output",
    )

    result = use_case.execute(job)

    assert storage.uploaded["bucket"] == "output"
    assert storage.uploaded["key"] == "clip.mp4"
    assert storage.uploaded["content"] == b"fake-video-bytes"
    assert result.bucket == "output"
    assert result.object_key == "clip.mp4"


def test_live_segment_header_is_cached_for_sequence_zero():
    header = b"\x1a\x45\xdf\xa3hdr" + b"\x1f\x43\xb6\x75"
    payload = header + b"\x01\x02"
    storage = FakeStorage([payload])
    cache = FakeHeaderCache()
    use_case = ProcessClipUseCase(
        storage=storage, processor=NoOpProcessor(), header_cache=cache
    )
    job = ClipJob(
        session_id="session",
        clip_id="clip123",
        source_bucket="input",
        object_key="clip.webm",
        target_bucket="output",
        metadata={
            "source": "live",
            "mime_type": "video/webm;codecs=vp9",
            "sequence_no": 0,
        },
    )

    use_case.execute(job)

    assert "session" in cache._storage
    assert cache._storage["session"].startswith(b"\x1a\x45\xdf\xa3")


def test_headerless_segment_is_patched_with_cached_header():
    header = b"\x1a\x45\xdf\xa3hdr" + b"\x1f\x43\xb6\x75"
    segment = b"\x1f\x43\xb6\x75" + b"\xaa\xbb"
    storage = FakeStorage([header + b"\x00", segment])
    cache = FakeHeaderCache()
    processor = NoOpProcessor()
    use_case = ProcessClipUseCase(
        storage=storage, processor=processor, header_cache=cache
    )
    job0 = ClipJob(
        session_id="session",
        clip_id="clip0",
        source_bucket="input",
        object_key="clip0.webm",
        target_bucket="output",
        metadata={
            "source": "live",
            "mime_type": "video/webm",
            "sequence_no": 0,
        },
    )
    job1 = ClipJob(
        session_id="session",
        clip_id="clip1",
        source_bucket="input",
        object_key="clip1.webm",
        target_bucket="output",
        metadata={
            "source": "live",
            "mime_type": "video/webm",
            "sequence_no": 1,
        },
    )

    use_case.execute(job0)
    use_case.execute(job1)

    assert storage.uploaded["content"].startswith(header)
