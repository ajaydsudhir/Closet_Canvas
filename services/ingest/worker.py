import threading
import traceback
from typing import Any, Optional
from pathlib import Path
import tempfile
import os
import json
import asyncio
import json
from pathlib import Path

from .application.use_cases.process_clip import ProcessClipUseCase
from .application.use_cases.gate_clip import GateClipUseCase
from .config import load_config, IngestConfig
from .domain.clip import ClipJob, GatingJob
from .infrastructure.storage import (
    create_s3_client as build_s3_client,
    create_storage_gateway,
)
from .infrastructure.remux import create_clip_processor
from .infrastructure.header_cache import create_header_cache
from .infrastructure.video_gating_service import create_gating_service
from .infrastructure.queue import (
    create_queue as build_queue,
    create_worker as build_worker,
)
from .infrastructure.status_publisher import SessionStatusPublisher
from .listener import listen_for_clip_ready
from .infrastructure.smpl_service import create_smpl_service

_CONFIG: IngestConfig | None = None
_USE_CASE: ProcessClipUseCase | None = None
_GATING_USE_CASE: GateClipUseCase | None = None
_RECOMMENDATION_USE_CASE: Any = None
_STATUS_PUBLISHER: SessionStatusPublisher | None = None


def update_status(session_id: str, status: str, message: str, progress: float):
    """Update session status via Redis pub/sub."""
    global _STATUS_PUBLISHER
    if _STATUS_PUBLISHER is None:
        cfg = get_config()
        _STATUS_PUBLISHER = SessionStatusPublisher(
            redis_host=cfg.redis_host,
            redis_port=cfg.redis_port,
        )
    try:
        asyncio.run(
            _STATUS_PUBLISHER.update_status(session_id, status, message, progress)
        )
    except Exception as e:
        print(f"[Worker] Failed to update status: {e}")


def get_config() -> IngestConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG


def create_s3_client():
    cfg = get_config()
    return build_s3_client(cfg)


def get_use_case() -> ProcessClipUseCase:
    global _USE_CASE
    if _USE_CASE is None:
        cfg = get_config()
        storage_gateway = create_storage_gateway(cfg)
        clip_processor = create_clip_processor(cfg)
        header_cache = create_header_cache(cfg)
        _USE_CASE = ProcessClipUseCase(
            storage=storage_gateway,
            processor=clip_processor,
            header_cache=header_cache,
        )
    return _USE_CASE


def get_gating_use_case() -> GateClipUseCase:
    """Factory function for gating use case with dependency injection."""
    global _GATING_USE_CASE
    if _GATING_USE_CASE is None:
        cfg = get_config()
        storage_gateway = create_storage_gateway(cfg)
        gating_service = create_gating_service()
        _GATING_USE_CASE = GateClipUseCase(
            storage=storage_gateway,
            gating_service=gating_service,
        )
    return _GATING_USE_CASE


def gate_video(
    session_id: str,
    clip_id: str,
    bucket_name: str,
    object_key: str,
    output_bucket: str = "smpl-gated",
    metadata: dict[str, object] | None = None,
) -> bool:
    update_status(session_id, "gating", "Checking video quality...", 33.0)

    use_case = get_gating_use_case()
    job = GatingJob(
        session_id=session_id,
        clip_id=clip_id,
        source_bucket=bucket_name,
        object_key=object_key,
        frames_bucket=output_bucket,
        metadata=metadata or {},
    )

    try:
        print(f"[Worker] Gating {bucket_name}/{object_key}")
        result = use_case.execute(job)
        print(
            f"[Worker] Gating result: passed={result.passed}, masks={len(result.masks)}"
        )
    except Exception:
        print(f"[Worker] Gating failed for {bucket_name}/{object_key}")
        traceback.print_exc()
        update_status(session_id, "error", "Video quality check failed", 0.0)
        return False

    if not result.passed:
        print(
            f"[Worker] Gated out {bucket_name}/{object_key} - reason: passed={result.passed}, masks={len(result.masks)}"
        )
        update_status(session_id, "error", "Video did not pass quality check", 0.0)
        return False

    print(
        f"[Worker] Passed gating: {len(result.masks)} masks generated for {bucket_name}/{object_key}"
    )

    # Save masks to MinIO
    mask_keys = []
    try:
        cfg = get_config()
        s3_client = create_s3_client()
        storage = create_storage_gateway(cfg)
        masks_bucket = "masks"

        # Ensure masks bucket exists
        try:
            s3_client.head_bucket(Bucket=masks_bucket)
        except Exception as _:
            s3_client.create_bucket(Bucket=masks_bucket)
            print(f"[Worker] Created bucket: {masks_bucket}")

        import numpy as np
        import cv2

        for idx, mask in enumerate(result.masks):
            # Create temporary file for mask
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                mask_path = tmp.name
                # Debug: check mask values
                person_pixels = int(mask.sum())
                total_pixels = mask.size
                coverage = (person_pixels / total_pixels) * 100
                print(
                    f"[Worker] Mask {idx}: shape={mask.shape}, dtype={mask.dtype}, person_pixels={person_pixels}/{total_pixels} ({coverage:.1f}% coverage)"
                )

                # Convert binary mask (0/1) to grayscale (0/255) for visibility
                if mask.dtype != np.uint8:
                    mask_uint8 = (mask * 255).astype(np.uint8)
                else:
                    # If already uint8 but values are 0/1, scale to 0/255
                    if mask.max() <= 1:
                        mask_uint8 = (mask * 255).astype(np.uint8)
                    else:
                        mask_uint8 = mask

                # Verify the output has proper values
                print(
                    f"[Worker] Mask {idx} after conversion: min={mask_uint8.min()}, max={mask_uint8.max()}"
                )
                cv2.imwrite(mask_path, mask_uint8)

                # Upload to MinIO
                mask_key = f"{result.session_id}/{result.clip_id}/mask_{idx:03d}.png"
                storage.upload(masks_bucket, mask_key, mask_path)
                mask_keys.append(mask_key)

                # Clean up temp file
                os.remove(mask_path)

        print(f"[Worker] Saved {len(mask_keys)} masks to {masks_bucket}")
    except Exception:
        print("[Worker] Failed to save masks to MinIO")
        traceback.print_exc()
        return False

    # Enqueue to SMPL queue with mask references
    try:
        enqueue_smpl(
            result.session_id,
            result.clip_id,
            bucket_name,
            object_key,
            masks_bucket,
            mask_keys,
        )
    except Exception:
        print("[Worker] Failed to enqueue to SMPL queue")
        traceback.print_exc()

    return True


def enqueue_gating(
    session_id: str,
    clip_id: str,
    bucket: str,
    key: str,
):
    queue = build_queue(get_config(), queue_name="gating")
    job = queue.enqueue(gate_video, session_id, clip_id, bucket, key, job_timeout=1500)
    print(
        f"Enqueued gating job id={getattr(job, 'id', 'unknown')} for session={session_id} clip={clip_id}"
    )
    return job


def process_smpl(
    session_id: str,
    clip_id: str,
    video_bucket: str,
    video_key: str,
    masks_bucket: str,
    mask_keys: list[str],
) -> bool:
    """Process SMPL estimation from video and masks."""
    update_status(session_id, "smpl", "Analyzing body pose...", 67.0)

    print(f"[SMPL Worker] Processing session={session_id} clip={clip_id}")
    print(f"[SMPL Worker] Video: {video_bucket}/{video_key}")
    print(f"[SMPL Worker] Masks: {masks_bucket} with {len(mask_keys)} masks")
    print(
        f"[SMPL Worker] Mask keys: {mask_keys[:3]}..."
        if len(mask_keys) > 3
        else f"[SMPL Worker] Mask keys: {mask_keys}"
    )

    try:
        cfg = get_config()
        storage = create_storage_gateway(cfg)
        s3_client = create_s3_client()

        # 1. Download video from storage
        print(f"[SMPL Worker] Downloading video from {video_bucket}/{video_key}")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
            video_path = Path(tmp_video.name)

        storage.download(video_bucket, video_key, str(video_path))
        print(f"[SMPL Worker] Downloaded video to {video_path}")

        # 2. Run SMPL estimation using service
        print("[SMPL Worker] Initializing body measurement service")
        smpl_service = create_smpl_service(user_height_cm=170.0)

        print("[SMPL Worker] Running SMPL estimation")
        measurements = smpl_service.estimate_from_video(video_path)

        update_status(session_id, "finishing", "Finalizing analysis...", 90.0)

        if "error" in measurements:
            print(f"[SMPL Worker] SMPL estimation failed: {measurements['error']}")
            video_path.unlink()
            return False

        print("[SMPL Worker] ✓ Body measurements estimated:")
        print(f"  Height: {measurements.get('height_cm')}cm")
        print(f"  Shoulder Width: {measurements.get('shoulder_width_cm'):.1f}cm")
        print(
            f"  Chest Circumference: {measurements.get('chest_circumference_cm'):.1f}cm"
        )
        print(
            f"  Waist Circumference: {measurements.get('waist_circumference_cm'):.1f}cm"
        )
        print(f"  Hip Circumference: {measurements.get('hip_circumference_cm'):.1f}cm")
        print(f"  Images Used: {measurements.get('images_used')}")

        # 3. Upload results to output bucket
        output_bucket = "smpl-measurements"

        # Ensure output bucket exists
        try:
            s3_client.head_bucket(Bucket=output_bucket)
        except Exception:
            s3_client.create_bucket(Bucket=output_bucket)
            print(f"[SMPL Worker] Created bucket: {output_bucket}")

        # Save measurements as JSON
        measurements_key = f"{session_id}/{clip_id}/measurements.json"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_json:
            json.dump(measurements, tmp_json, indent=2)
            json_path = Path(tmp_json.name)

        storage.upload(output_bucket, measurements_key, str(json_path))
        print(
            f"[SMPL Worker] Uploaded measurements to {output_bucket}/{measurements_key}"
        )

        json_path.unlink()
        video_path.unlink()

        print(f"[SMPL Worker] ✓ Completed SMPL processing for {session_id}/{clip_id}")

        # Enqueue recommendation processing
        try:
            enqueue_recommendation(session_id, clip_id)
        except Exception:
            print("[SMPL Worker] Failed to enqueue recommendation job")
            traceback.print_exc()

        return True

    except Exception as e:
        print(f"[SMPL Worker] ✗ SMPL processing failed: {str(e)}")
        traceback.print_exc()
        return False


def get_recommendation_use_case():
    """Factory function for recommendation use case with dependency injection."""
    # Lazy imports to avoid loading numpy in CPU worker
    from .application.use_cases.recommend import GenerateRecommendationsUseCase
    from .infrastructure.recommendation_service import GarmentCatalogService
    
    global _RECOMMENDATION_USE_CASE
    if _RECOMMENDATION_USE_CASE is None:
        cfg = get_config()
        storage_gateway = create_storage_gateway(cfg)
        catalog_service = GarmentCatalogService()
        _RECOMMENDATION_USE_CASE = GenerateRecommendationsUseCase(
            storage=storage_gateway,
            catalog_service=catalog_service,
        )
    return _RECOMMENDATION_USE_CASE


def process_recommendation(
    session_id: str,
    clip_id: str,
    user_id: str | None = None,
    categories: list[str] | None = None,
) -> bool:
    """Process recommendation generation from body measurements."""
    # Lazy import to avoid loading numpy in CPU worker
    from .application.use_cases.recommend import RecommendationJob
    
    update_status(session_id, "recommending", "Finding your perfect fits...", 85.0)

    print(f"[Recommendation Worker] Processing session={session_id} clip={clip_id}")

    try:
        use_case = get_recommendation_use_case()
        job = RecommendationJob(
            session_id=session_id,
            clip_id=clip_id,
            user_id=user_id,
            categories=categories,
            min_fit_score=25.0,
            limit=20,
        )

        result = use_case.execute(job)

        if not result.success:
            print(f"[Recommendation Worker] ✗ Failed: {result.error}")
            update_status(session_id, "error", f"Recommendation failed: {result.error}", 0.0)
            return False

        print(f"[Recommendation Worker] ✓ Generated {result.total_count} recommendations")

        # Store recommendations in Redis for frontend retrieval
        cfg = get_config()
        publisher = SessionStatusPublisher(
            redis_host=cfg.redis_host,
            redis_port=cfg.redis_port,
        )
        asyncio.run(publisher.store_recommendations(session_id, result.recommendations))

        # Update status to complete
        update_status(session_id, "complete", "Recommendations ready!", 100.0)

        print(f"[Recommendation Worker] ✓ Completed for {session_id}/{clip_id}")
        return True

    except Exception as e:
        print(f"[Recommendation Worker] ✗ Processing failed: {str(e)}")
        traceback.print_exc()
        update_status(session_id, "error", "Recommendation processing failed", 0.0)
        return False


def enqueue_recommendation(
    session_id: str,
    clip_id: str,
    user_id: str | None = None,
    categories: list[str] | None = None,
):
    """Enqueue a recommendation processing job."""
    queue = build_queue(get_config(), queue_name="recommendation")
    job = queue.enqueue(
        process_recommendation,
        session_id,
        clip_id,
        user_id,
        categories,
        job_timeout=600,
    )
    print(
        f"[Worker] Enqueued recommendation job id={getattr(job, 'id', 'unknown')} for session={session_id} clip={clip_id}"
    )
    return job


def enqueue_smpl(
    session_id: str,
    clip_id: str,
    video_bucket: str,
    video_key: str,
    masks_bucket: str,
    mask_keys: list[str],
):
    queue = build_queue(get_config(), queue_name="smpl")
    job = queue.enqueue(
        process_smpl,
        session_id,
        clip_id,
        video_bucket,
        video_key,
        masks_bucket,
        mask_keys,
        job_timeout=2000,
    )
    print(
        f"[Worker] Enqueued SMPL job id={getattr(job, 'id', 'unknown')} for session={session_id} clip={clip_id}"
    )
    return job


def process_video(
    session_id: str,
    clip_id: str,
    bucket_name: str,
    object_key: str,
    output_bucket: str = "processed",
    metadata: dict[str, object] | None = None,
):
    """Process a clip via the configured use case."""
    update_status(session_id, "recording", "Processing video...", 10.0)

    use_case = get_use_case()
    job = ClipJob(
        session_id=session_id,
        clip_id=clip_id,
        source_bucket=bucket_name,
        object_key=object_key,
        target_bucket=output_bucket or bucket_name,
        metadata=metadata or {},
    )
    print(f"[Worker] Processing {bucket_name}/{object_key}")
    use_case.execute(job)
    print(f"[Worker] completed {job.target_bucket}/{job.object_key}")

    try:
        enqueue_gating(job.session_id, job.clip_id, job.target_bucket, job.object_key)
    except Exception:
        print("[Worker] failed to enqueue gating job")
        traceback.print_exc()
        update_status(session_id, "error", "Failed to queue quality check", 0.0)


def enqueue(
    session_id: str,
    clip_id: str,
    bucket: str,
    key: str,
    output_bucket: str = "processed",
    metadata: dict[str, object] | None = None,
):
    queue = build_queue(get_config())
    job = queue.enqueue(
        process_video, session_id, clip_id, bucket, key, output_bucket, metadata or {}
    )
    print(f"Enqueued job id={job.id} for session={session_id} clip={clip_id}")
    return job


def run_worker(queue_name: Optional[str] = None):
    cfg = get_config()
    worker = build_worker(cfg, queue_name=queue_name)
    print(f"Starting worker for queue: {queue_name or cfg.redis_queue_name}")
    worker.work()


def run_worker_service(
    queue_name: Optional[str] = None,
    enable_listener: bool = True,
):
    stop_event = threading.Event()
    cfg = get_config()

    listener_thread = None
    if enable_listener:
        listener_thread = threading.Thread(
            target=listen_for_clip_ready,
            args=(
                cfg,
                lambda payload: enqueue(
                    payload["session_id"],
                    payload["clip_id"],
                    payload["bucket"],
                    payload["object_key"],
                    payload["bucket"],
                    payload.get("metadata") or {},
                ),
                stop_event,
            ),
            daemon=True,
        )
        listener_thread.start()
        print("Started listener thread")

    try:
        run_worker(queue_name=queue_name)
    except KeyboardInterrupt:
        print("\nShutting down worker...")
    finally:
        stop_event.set()
        if listener_thread and listener_thread.is_alive():
            listener_thread.join(timeout=2)
