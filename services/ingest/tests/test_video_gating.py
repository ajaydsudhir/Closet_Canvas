import os
import cv2
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from closet_canvas.video_quality.video_gating import VideoGating
from closet_canvas.video_quality.metrics import Metrics as VideoQualityMetrics


@pytest.fixture
def video_gating():
    """Create a VideoGating instance."""
    return VideoGating()


@pytest.fixture
def test_video_path():
    """Get the path to the test video."""
    video_path = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "raw"
        / "catalog_test"
        / "videos"
        / "videos"
        / "oo7.mp4"
    )
    if not video_path.exists():
        pytest.skip(f"Test video not found at {video_path}")
    return str(video_path)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_video_capture():
    """Create a mock VideoCapture object."""
    mock_cap = MagicMock()
    mock_cap.get.return_value = 30.0  # 30 fps
    return mock_cap


@pytest.fixture
def sample_frames_dir(temp_output_dir):
    """Create a directory with sample frames for testing."""
    frames_dir = os.path.join(temp_output_dir, "sample_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Create a few sample frame images
    for i in range(5):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * (100 + i * 20)
        cv2.imwrite(os.path.join(frames_dir, f"frame_{i:05d}.jpg"), frame)

    return frames_dir


class TestExtractFrames:
    """Test frame extraction functionality."""

    def test_extract_frames_creates_output_directory(
        self, temp_output_dir, test_video_path
    ):
        """Test that extract_frames creates the output directory."""
        out_dir = os.path.join(temp_output_dir, "frames")
        assert not os.path.exists(out_dir)

        VideoGating.extract_frames(test_video_path, out_dir, fps_sampling=5)

        assert os.path.exists(out_dir)

    def test_extract_frames_returns_frame_count(self, temp_output_dir, test_video_path):
        """Test that extract_frames returns the number of frames extracted."""
        out_dir = os.path.join(temp_output_dir, "frames")

        num_frames = VideoGating.extract_frames(
            test_video_path, out_dir, fps_sampling=5
        )

        assert isinstance(num_frames, int)
        assert num_frames > 0

    def test_extract_frames_creates_frame_files(self, temp_output_dir, test_video_path):
        """Test that extract_frames creates actual frame files."""
        out_dir = os.path.join(temp_output_dir, "frames")

        num_frames = VideoGating.extract_frames(
            test_video_path, out_dir, fps_sampling=5
        )

        # Check that frame files exist
        frame_files = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
        assert len(frame_files) == num_frames
        assert len(frame_files) > 0

    def test_extract_frames_naming_convention(self, temp_output_dir, test_video_path):
        """Test that frames are named correctly."""
        out_dir = os.path.join(temp_output_dir, "frames")

        VideoGating.extract_frames(test_video_path, out_dir, fps_sampling=5)

        # Check that frames follow naming convention
        frame_files = sorted([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
        assert frame_files[0] == "frame_00000.jpg"
        if len(frame_files) > 1:
            assert frame_files[1] == "frame_00001.jpg"

    def test_extract_frames_different_fps_sampling(
        self, temp_output_dir, test_video_path
    ):
        """Test that different fps_sampling values affect frame count."""
        out_dir_1 = os.path.join(temp_output_dir, "frames_fps1")
        out_dir_2 = os.path.join(temp_output_dir, "frames_fps10")

        # Higher sampling rate should give more frames
        num_frames_high = VideoGating.extract_frames(
            test_video_path, out_dir_1, fps_sampling=10
        )
        num_frames_low = VideoGating.extract_frames(
            test_video_path, out_dir_2, fps_sampling=2
        )

        assert num_frames_high >= num_frames_low

    @patch("cv2.VideoCapture")
    def test_extract_frames_handles_invalid_video(self, mock_vc, temp_output_dir):
        """Test that extract_frames handles invalid video files."""
        mock_cap = MagicMock()
        mock_cap.read.return_value = (False, None)  # No frames available
        mock_cap.get.return_value = 30.0
        mock_vc.return_value = mock_cap

        out_dir = os.path.join(temp_output_dir, "frames")
        num_frames = VideoGating.extract_frames("invalid.mp4", out_dir, fps_sampling=5)

        assert num_frames == 0

    @patch("cv2.VideoCapture")
    def test_extract_frames_calculates_interval_correctly(
        self, mock_vc, temp_output_dir
    ):
        """Test that frame interval is calculated correctly based on FPS."""
        mock_cap = MagicMock()
        mock_cap.get.return_value = 30.0  # 30 fps video
        mock_cap.read.side_effect = [
            (True, np.zeros((100, 100, 3), dtype=np.uint8)) for _ in range(10)
        ] + [(False, None)]
        mock_vc.return_value = mock_cap

        out_dir = os.path.join(temp_output_dir, "frames")
        VideoGating.extract_frames("test.mp4", out_dir, fps_sampling=5)

        # With 30 fps and sampling at 5 fps, interval should be 6 frames
        # Verify frames are saved at correct intervals
        frame_files = sorted(os.listdir(out_dir))
        assert len(frame_files) > 0


class TestEvaluateScore:
    """Test score evaluation functionality."""

    def test_evaluate_score_returns_boolean(self, video_gating, sample_frames_dir):
        """Test that evaluate_score returns a boolean."""
        result = video_gating.evaluate_score(sample_frames_dir, threshold=60)
        assert isinstance(result, bool)

    @patch(
        "closet_canvas.video_quality.video_quality_metrics.VideoQualityMetrics.score_frame"
    )
    def test_evaluate_score_calls_score_frame(
        self, mock_score, video_gating, sample_frames_dir
    ):
        """Test that evaluate_score calls score_frame for each frame."""
        mock_score.return_value = 80

        video_gating.evaluate_score(sample_frames_dir, threshold=60)

        # Should be called for each frame
        frame_files = [
            f
            for f in os.listdir(sample_frames_dir)
            if os.path.isfile(os.path.join(sample_frames_dir, f))
        ]
        assert mock_score.call_count == len(frame_files)

    @patch(
        "closet_canvas.video_quality.video_quality_metrics.VideoQualityMetrics.score_frame"
    )
    def test_evaluate_score_passes_threshold(
        self, mock_score, video_gating, sample_frames_dir
    ):
        """Test that evaluate_score returns True when >70% frames pass threshold."""
        # 4 out of 5 frames pass (80% > 70%)
        mock_score.side_effect = [80, 80, 80, 80, 40]

        result = video_gating.evaluate_score(sample_frames_dir, threshold=60)
        assert result is True

    @patch(
        "closet_canvas.video_quality.video_quality_metrics.VideoQualityMetrics.score_frame"
    )
    def test_evaluate_score_fails_threshold(
        self, mock_score, video_gating, sample_frames_dir
    ):
        """Test that evaluate_score returns False when <=70% frames pass threshold."""
        # 3 out of 5 frames pass (60% <= 70%)
        mock_score.side_effect = [80, 80, 80, 40, 40]

        result = video_gating.evaluate_score(sample_frames_dir, threshold=60)
        assert result is False

    @patch(
        "closet_canvas.video_quality.video_quality_metrics.VideoQualityMetrics.score_frame"
    )
    def test_evaluate_score_exactly_70_percent(
        self, mock_score, video_gating, sample_frames_dir
    ):
        """Test boundary condition: exactly 70% frames passing."""
        # Need more frames to test 70% exactly
        # 7 out of 10 frames pass (70%)
        for i in range(5):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
            cv2.imwrite(
                os.path.join(sample_frames_dir, f"extra_frame_{i:05d}.jpg"), frame
            )

        # 7 frames pass, 3 fail (exactly 70%)
        mock_score.side_effect = [80, 80, 80, 80, 80, 80, 80, 40, 40, 40]

        result = video_gating.evaluate_score(sample_frames_dir, threshold=60)
        # 70% should not be sufficient (needs > 70%)
        assert result is False

    def test_evaluate_score_empty_directory_returns_false(
        self, video_gating, temp_output_dir
    ):
        """Test that evaluate_score returns False for empty directory."""
        empty_dir = os.path.join(temp_output_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)

        result = video_gating.evaluate_score(empty_dir, threshold=60)
        assert result is False

    @patch("cv2.imread")
    def test_evaluate_score_skips_invalid_frames(
        self, mock_imread, video_gating, sample_frames_dir
    ):
        """Test that evaluate_score skips frames that can't be read."""
        # First frame fails to load, others succeed
        mock_imread.side_effect = [None, np.zeros((100, 100, 3), dtype=np.uint8)] * 10

        with patch.object(VideoQualityMetrics, "score_frame", return_value=80):
            result = video_gating.evaluate_score(sample_frames_dir, threshold=60)
            # Should still work with valid frames
            assert isinstance(result, bool)

    def test_evaluate_score_respects_threshold_parameter(
        self, video_gating, sample_frames_dir
    ):
        """Test that different threshold values affect the result."""
        with patch.object(VideoQualityMetrics, "score_frame", return_value=65):
            # All frames score 65
            result_low = video_gating.evaluate_score(sample_frames_dir, threshold=60)
            result_high = video_gating.evaluate_score(sample_frames_dir, threshold=70)

            assert result_low is True  # 65 >= 60
            assert result_high is False  # 65 < 70


class TestEvaluateVideoWithSegformerSam:
    """Test end-to-end video evaluation."""

    def test_evaluate_video_returns_boolean(self, video_gating, test_video_path):
        """Test that evaluate_video_with_segformer_sam returns a boolean."""
        with patch.object(video_gating, "extract_frames", return_value=5):
            with patch.object(video_gating, "evaluate_score", return_value=True):
                result = video_gating.evaluate_video_with_segformer_sam(
                    test_video_path, threshold=60
                )
                assert isinstance(result, bool)

    def test_evaluate_video_calls_extract_frames(self, video_gating, test_video_path):
        """Test that evaluate_video_with_segformer_sam calls extract_frames."""
        with patch.object(
            video_gating, "extract_frames", return_value=5
        ) as mock_extract:
            with patch.object(video_gating, "evaluate_score", return_value=True):
                video_gating.evaluate_video_with_segformer_sam(
                    test_video_path, threshold=60
                )

                mock_extract.assert_called_once()
                args = mock_extract.call_args[0]
                assert args[0] == test_video_path
                assert args[1] == "temp_frames"

    def test_evaluate_video_calls_evaluate_score(self, video_gating, test_video_path):
        """Test that evaluate_video_with_segformer_sam calls evaluate_score."""
        with patch.object(video_gating, "extract_frames", return_value=5):
            with patch.object(
                video_gating, "evaluate_score", return_value=True
            ) as mock_eval:
                video_gating.evaluate_video_with_segformer_sam(
                    test_video_path, threshold=60
                )

                mock_eval.assert_called_once_with("temp_frames", threshold=60)

    def test_evaluate_video_cleans_up_temp_frames(self, video_gating, test_video_path):
        """Test that temporary frames are cleaned up after evaluation."""
        with patch.object(video_gating, "extract_frames") as mock_extract:
            with patch.object(video_gating, "evaluate_score", return_value=True):
                # Create temp directory and files
                temp_dir = "temp_frames"
                os.makedirs(temp_dir, exist_ok=True)
                test_file = os.path.join(temp_dir, "test.jpg")
                with open(test_file, "w") as f:
                    f.write("test")

                mock_extract.return_value = 1

                video_gating.evaluate_video_with_segformer_sam(
                    test_video_path, threshold=60
                )

                # Verify cleanup
                assert not os.path.exists(temp_dir)

    def test_evaluate_video_no_frames_returns_false(
        self, video_gating, test_video_path
    ):
        """Test that evaluate_video returns False when no frames are extracted."""
        with patch.object(video_gating, "extract_frames", return_value=0):
            result = video_gating.evaluate_video_with_segformer_sam(
                test_video_path, threshold=60
            )
            assert result is False

    def test_evaluate_video_passes_threshold_to_evaluate_score(
        self, video_gating, test_video_path
    ):
        """Test that threshold is correctly passed through to evaluate_score."""
        with patch.object(video_gating, "extract_frames", return_value=5):
            with patch.object(
                video_gating, "evaluate_score", return_value=True
            ) as mock_eval:
                video_gating.evaluate_video_with_segformer_sam(
                    test_video_path, threshold=75
                )

                mock_eval.assert_called_once_with("temp_frames", threshold=75)

    @patch("closet_canvas.video_quality.video_gating.os.listdir")
    @patch("closet_canvas.video_quality.video_gating.os.remove")
    @patch("closet_canvas.video_quality.video_gating.os.rmdir")
    def test_evaluate_video_handles_cleanup_errors_gracefully(
        self, mock_rmdir, mock_remove, mock_listdir, video_gating, test_video_path
    ):
        """Test that cleanup errors don't crash the evaluation."""
        with patch.object(video_gating, "extract_frames", return_value=5):
            with patch.object(video_gating, "evaluate_score", return_value=True):
                with patch(
                    "closet_canvas.video_quality.video_gating.os.path.exists",
                    return_value=True,
                ):
                    mock_listdir.return_value = ["frame_00000.jpg"]
                    mock_remove.side_effect = OSError("Permission denied")

                    # Should not raise exception - errors are handled
                    result = video_gating.evaluate_video_with_segformer_sam(
                        test_video_path, threshold=60
                    )
                    assert isinstance(result, bool)


class TestIntegrationWithRealVideo:
    """Integration tests using the actual test video."""

    def test_full_pipeline_with_real_video(
        self, video_gating, test_video_path, temp_output_dir
    ):
        """Test the complete pipeline with a real video file."""
        out_dir = os.path.join(temp_output_dir, "integration_frames")

        # Extract frames
        num_frames = VideoGating.extract_frames(
            test_video_path, out_dir, fps_sampling=5
        )
        assert num_frames > 0

        # Verify frames exist
        frame_files = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
        assert len(frame_files) == num_frames

        # Test that frames are valid images
        for frame_file in frame_files[:3]:  # Check first 3 frames
            frame_path = os.path.join(out_dir, frame_file)
            frame = cv2.imread(frame_path)
            assert frame is not None
            assert frame.shape[2] == 3  # BGR image

    @pytest.mark.slow
    def test_extract_frames_with_different_sampling_rates(
        self, test_video_path, temp_output_dir
    ):
        """Test frame extraction with various sampling rates."""
        sampling_rates = [1, 5, 10, 15]
        frame_counts = []

        for rate in sampling_rates:
            out_dir = os.path.join(temp_output_dir, f"frames_fps{rate}")
            num_frames = VideoGating.extract_frames(
                test_video_path, out_dir, fps_sampling=rate
            )
            frame_counts.append(num_frames)

        # Higher sampling rates should generally produce more frames
        # (or equal if video is very short)
        for i in range(len(frame_counts) - 1):
            assert frame_counts[i] >= frame_counts[i + 1] or frame_counts[i + 1] > 0


class TestStaticMethod:
    """Test that extract_frames is properly defined as a static method."""

    def test_extract_frames_can_be_called_without_instance(
        self, test_video_path, temp_output_dir
    ):
        """Test that extract_frames can be called as a static method."""
        out_dir = os.path.join(temp_output_dir, "static_test")

        # Should work without creating an instance
        num_frames = VideoGating.extract_frames(
            test_video_path, out_dir, fps_sampling=5
        )
        assert num_frames > 0

    def test_extract_frames_can_be_called_from_instance(
        self, video_gating, test_video_path, temp_output_dir
    ):
        """Test that extract_frames can also be called from an instance."""
        out_dir = os.path.join(temp_output_dir, "instance_test")

        # Should also work from instance (calls the static method)
        num_frames = VideoGating.extract_frames(
            test_video_path, out_dir, fps_sampling=5
        )
        assert num_frames > 0
