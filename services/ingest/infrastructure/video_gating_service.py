from __future__ import annotations

from pathlib import Path

from services.ingest.application.interfaces import VideoGatingService


class SegformerSAMGatingService(VideoGatingService):
    """
    Video quality gating service using Segformer and SAM.
    Evaluates video quality and generates person segmentation masks.
    """

    def evaluate(self, video_path: Path) -> tuple[bool, list]:
        """
        Evaluate video quality and generate masks.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (allowed: bool, masks: list)
            - allowed: True if video passes quality thresholds
            - masks: List of segmentation masks from frames
        """
        try:
            from closet_canvas.video_quality.video_gating import VideoGating
        except ImportError as e:
            print(f"[GatingService] Failed to import VideoGating: {e}")
            return False, []

        try:
            gating = VideoGating()
            allowed, masks = gating.evaluate_video_with_segformer_sam(
                video_path.as_posix()
            )
            print(
                f"[GatingService] Evaluation complete: allowed={allowed}, masks={len(masks)}"
            )
            return allowed, masks
        except Exception as e:
            import traceback

            print(f"[GatingService] Exception during gating: {e}")
            traceback.print_exc()
            return False, []


def create_gating_service() -> VideoGatingService:
    """Factory function to create video gating service instance."""
    return SegformerSAMGatingService()
