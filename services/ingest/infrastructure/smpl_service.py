from __future__ import annotations

import asyncio
from pathlib import Path


class SMPLService:
    """
    Body measurement estimation service using SMPL generator.
    Estimates body measurements from video using pose detection.
    """

    def __init__(self, user_height_cm: float = 170.0):
        """
        Initialize SMPL service.

        Args:
            user_height_cm: User height in centimeters (default 170cm average adult)
        """
        self.user_height_cm = user_height_cm

    def estimate_from_video(self, video_path: Path) -> dict:
        """
        Estimate body measurements from video.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary containing body measurements and metadata
        """
        try:
            from closet_canvas.smpl_generator.smpl import BodyMeasurementEstimatorAsync
        except ImportError as e:
            print(f"[SMPLService] Failed to import dependencies: {e}")
            return {"error": f"Missing dependencies: {e}"}

        try:
            # Extract frames from video
            print(f"[SMPLService] Extracting frames from video: {video_path}")
            frames = self._extract_frames(video_path)

            if not frames:
                return {"error": "No frames extracted from video"}

            print(f"[SMPLService] Extracted {len(frames)} frames")

            # Initialize estimator and run async estimation
            estimator = BodyMeasurementEstimatorAsync(
                model_complexity=2,
                min_detection_confidence=0.5,
                max_concurrency=4,
            )

            print(f"[SMPLService] Running pose estimation on {len(frames)} frames")
            measurements = asyncio.run(estimator.estimate(frames, self.user_height_cm))

            return measurements

        except Exception as e:
            import traceback

            print(f"[SMPLService] Exception during estimation: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    @staticmethod
    def _extract_frames(video_path: Path, sample_size: int = 10) -> list:
        """
        Extract frames from video.

        Args:
            video_path: Path to video file
            sample_size: Number of frames to extract (evenly spaced)

        Returns:
            List of OpenCV frames (numpy arrays)
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python is required")

        frames = []
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count == 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")

        # Sample every Nth frame
        sample_interval = max(1, frame_count // sample_size)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                frames.append(frame)

            frame_idx += 1

        cap.release()
        return frames


def create_smpl_service(user_height_cm: float = 170.0) -> SMPLService:
    """Factory function to create SMPL service instance."""
    return SMPLService(user_height_cm=user_height_cm)
