from unittest.mock import MagicMock
import pytest

from services.ingest import worker
from services.ingest.domain.clip import ClipJob


def test_process_video_roundtrip(monkeypatch):
    captured = {}

    class FakeUseCase:
        def execute(self, job: ClipJob):
            captured["job"] = job

    monkeypatch.setattr(worker, "get_use_case", lambda: FakeUseCase())

    worker.process_video(
        "session", "clip123", "input-bucket", "clip.mp4", output_bucket="out-bucket"
    )

    job = captured["job"]
    assert job.source_bucket == "input-bucket"
    assert job.target_bucket == "out-bucket"
    assert job.object_key == "clip.mp4"


def test_enqueue_calls_queue(monkeypatch):
    fake_queue = MagicMock()
    fake_job = MagicMock()
    fake_job.id = "job-123"
    fake_queue.enqueue.return_value = fake_job

    class FakeConfig:
        redis_queue_name = "video"
        redis_host = "localhost"
        redis_port = 6379
        redis_db = 0
        storage_endpoint_url = ""
        storage_region = ""
        storage_access_key = ""
        storage_secret_key = ""
        redis_channel = "clip_ready"
        remux_target_extension = ".mp4"
        header_cache_ttl_seconds = 3600

    monkeypatch.setattr(worker, "get_config", lambda: FakeConfig())
    monkeypatch.setattr(worker, "build_queue", lambda cfg: fake_queue)

    job = worker.enqueue(
        "sess", "clip", "my-bucket", "file.mov", output_bucket="processed"
    )

    fake_queue.enqueue.assert_called_once_with(
        worker.process_video, "sess", "clip", "my-bucket", "file.mov", "processed", {}
    )
    assert job.id == "job-123"


def test_create_s3_client_respects_env(monkeypatch):
    # Set environment vars and intercept boto3.client calls
    monkeypatch.setenv("MINIO_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("MINIO_REGION", "eu-west-1")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "AK")
    monkeypatch.setenv("MINIO_SECRET_KEY", "SK")

    called = {}

    def fake_builder(cfg):
        called["cfg"] = cfg
        return MagicMock()

    monkeypatch.setattr(worker, "build_s3_client", fake_builder)

    # call create_s3_client which should use the patched builder
    monkeypatch.setattr(worker, "_CONFIG", None)
    client = worker.create_s3_client()

    cfg = called["cfg"]
    assert cfg.storage_endpoint_url == "http://minio:9000"
    assert cfg.storage_region == "eu-west-1"
    assert cfg.storage_access_key == "AK"
    assert cfg.storage_secret_key == "SK"
    assert client is not None


def test_process_video_failure_bubbles(monkeypatch):
    class FakeUseCase:
        def execute(self, job):
            raise RuntimeError("download failed")

    monkeypatch.setattr(worker, "get_use_case", lambda: FakeUseCase())

    with pytest.raises(RuntimeError):
        worker.process_video("sess", "clip", "bucket", "key.mp4", output_bucket="out")


def test_run_worker_invokes_work(monkeypatch):
    called = {}

    def fake_build_worker(redis_cfg, queue_name=None):
        called["queue_name"] = queue_name

        class _Worker:
            def work(self_inner):
                called["worked"] = True

        called["redis_cfg"] = redis_cfg
        return _Worker()

    class FakeConfig:
        redis_queue_name = "video"
        redis_host = "localhost"
        redis_port = 6379
        redis_db = 0
        storage_endpoint_url = ""
        storage_region = ""
        storage_access_key = ""
        storage_secret_key = ""
        redis_channel = "clip_ready"
        remux_target_extension = ".mp4"
        header_cache_ttl_seconds = 3600

    fake_cfg = FakeConfig()
    monkeypatch.setattr(worker, "get_config", lambda: fake_cfg)
    monkeypatch.setattr(worker, "build_worker", fake_build_worker)

    worker.run_worker(queue_name="video")

    assert called.get("worked")
    assert called.get("queue_name") == "video"
