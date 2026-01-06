import numpy as np
import cv2
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from closet_canvas.video_quality.metrics import Metrics as VideoQualityMetrics


@pytest.fixture
def vqm():
    """Create a VideoQualityMetrics instance."""
    return VideoQualityMetrics()


@pytest.fixture
def sample_frame():
    """Create a sample BGR frame for testing."""
    # Create a 100x100 BGR image with some variation
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Add some content to make it realistic
    frame[20:80, 20:80] = [100, 150, 200]  # BGR values
    frame[40:60, 40:60] = [200, 200, 200]  # Brighter center
    return frame


@pytest.fixture
def dark_frame():
    """Create a dark frame for testing low lighting."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 30


@pytest.fixture
def bright_frame():
    """Create a bright frame for testing good lighting."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 150


@pytest.fixture
def blurry_frame():
    """Create a blurry frame for testing sharpness."""
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
    # Apply Gaussian blur to reduce sharpness
    return cv2.GaussianBlur(frame, (15, 15), 0)


@pytest.fixture
def sharp_frame():
    """Create a sharp frame with high contrast edges."""
    frame = np.ones((100, 100, 3), dtype=np.uint8) * 100
    # Add sharp edges
    frame[45:55, :] = 255  # Horizontal line
    frame[:, 45:55] = 0  # Vertical line
    return frame


@pytest.fixture
def sample_mask():
    """Create a sample binary mask."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1  # Person region
    return mask


class TestLightingMeasurement:
    """Test lighting measurement functionality."""

    def test_measure_lighting_returns_numeric(self, vqm, sample_frame):
        """Test that measure_lighting returns a numeric value."""
        lighting = vqm.measure_lighting(sample_frame)
        assert isinstance(lighting, (int, float, np.number))
        assert lighting >= 0

    def test_dark_frame_has_low_lighting(self, vqm, dark_frame):
        """Test that dark frames have low lighting values."""
        lighting = vqm.measure_lighting(dark_frame)
        assert lighting < 80

    def test_bright_frame_has_high_lighting(self, vqm, bright_frame):
        """Test that bright frames have high lighting values."""
        lighting = vqm.measure_lighting(bright_frame)
        assert lighting > 80

    def test_lighting_comparison(self, vqm, dark_frame, bright_frame):
        """Test that bright frames have higher lighting than dark frames."""
        dark_lighting = vqm.measure_lighting(dark_frame)
        bright_lighting = vqm.measure_lighting(bright_frame)
        assert bright_lighting > dark_lighting


class TestContrastMeasurement:
    """Test contrast measurement functionality."""

    def test_measure_contrast_returns_numeric(self, vqm, sample_frame):
        """Test that measure_contrast returns a numeric value."""
        contrast = vqm.measure_contrast(sample_frame)
        assert isinstance(contrast, (int, float, np.number))
        assert contrast >= 0

    def test_uniform_frame_has_low_contrast(self, vqm):
        """Test that uniform frames have low contrast."""
        uniform_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        contrast = vqm.measure_contrast(uniform_frame)
        assert contrast < 5  # Very low contrast for uniform frame

    def test_high_contrast_frame(self, vqm):
        """Test that high contrast frames have high contrast values."""
        high_contrast = np.zeros((100, 100, 3), dtype=np.uint8)
        high_contrast[::2, :] = 255  # Alternating black and white rows
        contrast = vqm.measure_contrast(high_contrast)
        assert contrast > 30


class TestSharpnessMeasurement:
    """Test sharpness measurement functionality."""

    def test_measure_sharpness_returns_numeric(self, vqm, sample_frame):
        """Test that measure_sharpness returns a numeric value."""
        sharpness = vqm.measure_sharpness(sample_frame)
        assert isinstance(sharpness, (int, float, np.number))
        assert sharpness >= 0

    def test_blurry_frame_has_low_sharpness(self, vqm, blurry_frame):
        """Test that blurry frames have low sharpness."""
        sharpness = vqm.measure_sharpness(blurry_frame)
        assert sharpness < 100

    def test_sharp_frame_has_high_sharpness(self, vqm, sharp_frame):
        """Test that sharp frames have high sharpness."""
        sharpness = vqm.measure_sharpness(sharp_frame)
        assert sharpness > 100

    def test_sharpness_comparison(self, vqm, blurry_frame, sharp_frame):
        """Test that sharp frames have higher sharpness than blurry frames."""
        blurry_sharpness = vqm.measure_sharpness(blurry_frame)
        sharp_sharpness = vqm.measure_sharpness(sharp_frame)
        assert sharp_sharpness > blurry_sharpness


class TestVisibilityScore:
    """Test visibility score functionality."""

    def test_visibility_score_returns_ratio(self, vqm, sample_frame):
        """Test that visibility_score returns a ratio between 0 and 1."""
        box = [20, 20, 80, 80]
        score = vqm.visibility_score(sample_frame, box)
        assert 0 <= score <= 1

    def test_full_frame_box_returns_one(self, vqm, sample_frame):
        """Test that a box covering the full frame returns score of 1."""
        h, w, _ = sample_frame.shape
        box = [0, 0, w, h]
        score = vqm.visibility_score(sample_frame, box)
        assert abs(score - 1.0) < 0.01

    def test_small_box_returns_small_score(self, vqm, sample_frame):
        """Test that a small box returns a small score."""
        box = [45, 45, 55, 55]  # 10x10 box in 100x100 frame
        score = vqm.visibility_score(sample_frame, box)
        assert score < 0.02  # Should be around 0.01


class TestPersonVisibilityScore:
    """Test person visibility score functionality."""

    def test_person_visibility_score_returns_ratio(self, vqm, sample_mask):
        """Test that person_visibility_score returns a ratio between 0 and 1."""
        score = vqm.person_visibility_score(sample_mask)
        assert 0 <= score <= 1

    def test_empty_mask_returns_zero(self, vqm):
        """Test that an empty mask returns score of 0."""
        empty_mask = np.zeros((100, 100), dtype=np.uint8)
        score = vqm.person_visibility_score(empty_mask)
        assert score == 0

    def test_full_mask_returns_one(self, vqm):
        """Test that a full mask returns score of 1."""
        full_mask = np.ones((100, 100), dtype=np.uint8)
        score = vqm.person_visibility_score(full_mask)
        assert score == 1

    def test_half_mask_returns_half(self, vqm):
        """Test that a half-filled mask returns score around 0.5."""
        half_mask = np.zeros((100, 100), dtype=np.uint8)
        half_mask[:50, :] = 1
        score = vqm.person_visibility_score(half_mask)
        assert abs(score - 0.5) < 0.01


class TestBodyCutoffScore:
    """Test body cutoff score functionality."""

    def test_centered_box_returns_one(self, vqm, sample_frame):
        """Test that a centered box returns 1 (no cutoff)."""
        box = [20, 20, 80, 80]
        score = vqm.body_cutoff_score(box, sample_frame)
        assert score == 1

    def test_left_edge_cutoff_returns_zero(self, vqm, sample_frame):
        """Test that a box touching left edge returns 0."""
        box = [2, 20, 80, 80]
        score = vqm.body_cutoff_score(box, sample_frame)
        assert score == 0

    def test_top_edge_cutoff_returns_zero(self, vqm, sample_frame):
        """Test that a box touching top edge returns 0."""
        box = [20, 2, 80, 80]
        score = vqm.body_cutoff_score(box, sample_frame)
        assert score == 0

    def test_right_edge_cutoff_returns_zero(self, vqm, sample_frame):
        """Test that a box touching right edge returns 0."""
        h, w, _ = sample_frame.shape
        box = [20, 20, w - 2, 80]
        score = vqm.body_cutoff_score(box, sample_frame)
        assert score == 0

    def test_bottom_edge_cutoff_returns_zero(self, vqm, sample_frame):
        """Test that a box touching bottom edge returns 0."""
        h, w, _ = sample_frame.shape
        box = [20, 20, 80, h - 2]
        score = vqm.body_cutoff_score(box, sample_frame)
        assert score == 0


class TestScoreFrame:
    """Test frame scoring functionality."""

    @patch("closet_canvas.video_quality.video_quality_metrics.Segmentation")
    def test_score_frame_returns_numeric(
        self, mock_segmentation_class, vqm, sample_frame
    ):
        """Test that score_frame returns a numeric score."""
        # Mock the segmentation instance and its methods
        mock_seg_instance = MagicMock()
        mock_segmentation_class.return_value = mock_seg_instance

        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_mask[20:80, 20:80] = 1

        mock_seg_instance.segment_person.return_value = mock_mask
        mock_seg_instance.refine_with_sam.return_value = mock_mask

        # Recreate VQM instance with mocked Segmentation
        vqm = VideoQualityMetrics()

        score = vqm.score_frame(sample_frame)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100

    @patch("closet_canvas.video_quality.video_quality_metrics.Segmentation")
    def test_score_frame_no_detection_returns_zero(
        self, mock_segmentation_class, vqm, sample_frame
    ):
        """Test that score_frame returns 0 when no person is detected."""
        mock_seg_instance = MagicMock()
        mock_segmentation_class.return_value = mock_seg_instance

        mock_seg_instance.segment_person.return_value = np.zeros(
            (100, 100), dtype=np.uint8
        )
        mock_seg_instance.refine_with_sam.return_value = None

        vqm = VideoQualityMetrics()

        score = vqm.score_frame(sample_frame)
        assert score == 0

    @patch("closet_canvas.video_quality.video_quality_metrics.Segmentation")
    def test_score_frame_good_conditions_high_score(self, mock_segmentation_class, vqm):
        """Test that good conditions result in a high score."""
        # Create ideal frame: bright, sharp, high contrast
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 120
        frame[40:60, :] = 255  # Sharp edges
        frame[:, 40:60] = 50  # High contrast

        mock_seg_instance = MagicMock()
        mock_segmentation_class.return_value = mock_seg_instance

        # Mock large visible person
        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_mask[10:90, 10:90] = 1  # Large person

        mock_seg_instance.segment_person.return_value = mock_mask
        mock_seg_instance.refine_with_sam.return_value = mock_mask

        vqm = VideoQualityMetrics()

        score = vqm.score_frame(frame)
        assert score > 50  # Should get points for multiple criteria

    @patch("closet_canvas.video_quality.video_quality_metrics.Segmentation")
    def test_score_frame_poor_visibility_low_score(
        self, mock_segmentation_class, vqm, sample_frame
    ):
        """Test that poor person visibility results in lower score."""
        mock_seg_instance = MagicMock()
        mock_segmentation_class.return_value = mock_seg_instance

        # Mock small person (low visibility)
        mock_mask = np.zeros((100, 100), dtype=np.uint8)
        mock_mask[45:55, 45:55] = 1  # Very small person

        mock_seg_instance.segment_person.return_value = mock_mask
        mock_seg_instance.refine_with_sam.return_value = mock_mask

        vqm = VideoQualityMetrics()

        score = vqm.score_frame(sample_frame)
        # Should not get the visibility points (40 points)
        assert score < 80

    @patch("closet_canvas.video_quality.video_quality_metrics.Segmentation")
    def test_score_frame_cutoff_penalty(
        self, mock_segmentation_class, vqm, sample_frame
    ):
        """Test that body cutoff results in score penalty."""
        mock_seg_instance = MagicMock()
        mock_segmentation_class.return_value = mock_seg_instance

        # Mock person at edge (cutoff) - mask extends to edge
        mock_mask_cutoff = np.zeros((100, 100), dtype=np.uint8)
        mock_mask_cutoff[0:80, 20:80] = 1  # Touches top edge

        mock_seg_instance.segment_person.return_value = mock_mask_cutoff
        mock_seg_instance.refine_with_sam.return_value = mock_mask_cutoff

        vqm = VideoQualityMetrics()
        score = vqm.score_frame(sample_frame)

        # Now test with no cutoff - centered person
        mock_mask_no_cutoff = np.zeros((100, 100), dtype=np.uint8)
        mock_mask_no_cutoff[20:80, 20:80] = 1  # Centered, no edge touching

        mock_seg_instance.segment_person.return_value = mock_mask_no_cutoff
        mock_seg_instance.refine_with_sam.return_value = mock_mask_no_cutoff

        vqm = VideoQualityMetrics()
        score_no_cutoff = vqm.score_frame(sample_frame)

        # Score without cutoff should be higher (15 points difference)
        assert score_no_cutoff >= score


class TestIntegration:
    """Integration tests with real data."""

    def test_all_metrics_on_real_frame(self, vqm):
        """Test all metrics work together on a realistic frame."""
        # Create a realistic frame
        frame = cv2.imread(
            str(
                Path(__file__).parent.parent.parent
                / "data"
                / "raw"
                / "catalog_test"
                / "fashionpedia_subset"
                / "0.jpg"
            )
        )

        if frame is not None:
            lighting = vqm.measure_lighting(frame)
            contrast = vqm.measure_contrast(frame)
            sharpness = vqm.measure_sharpness(frame)

            assert lighting >= 0
            assert contrast >= 0
            assert sharpness >= 0

            # All should be reasonable values
            assert 0 <= lighting <= 255
            assert contrast < 200  # Standard deviation shouldn't exceed typical range
