import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image

from closet_canvas.catalog import segmentation


# Path to test images
TEST_IMAGE_DIR = (
    Path(__file__).parent.parent.parent
    / "data"
    / "raw"
    / "catalog_test"
    / "fashionpedia_subset"
)


def _make_dummy_feature_extractor_and_segformer(image_h=10, image_w=10, num_classes=20):
    class DummyInputs(dict):
        def __init__(self, pixel_values):
            # store as a mapping so it can be unpacked with **inputs
            super().__init__({"pixel_values": pixel_values})

        def to(self, device):
            # mimic transformers BatchEncoding.to(device)
            self["pixel_values"] = self["pixel_values"].to(device)
            return self

    class DummyFeatureExtractor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def __call__(self, images=None, return_tensors=None):
            pix = torch.zeros(1, 3, image_h, image_w)
            return DummyInputs(pix)

    class DummySegformerOutput:
        def __init__(self, logits):
            self.logits = logits

    class DummySegformer:
        def __init__(self):
            self._param = torch.nn.Parameter(torch.zeros(1))

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return self

        def to(self, device):
            self._param.data = self._param.data.to(device)
            return self

        def parameters(self):
            return iter([self._param])

        def __call__(self, **inputs):
            device = next(self.parameters()).device
            # Create logits with low resolution that will be upsampled
            # Using smaller dims so interpolate will upsample to target size
            small_h = max(1, image_h // 4)
            small_w = max(1, image_w // 4)
            logits = torch.zeros(1, num_classes, small_h, small_w, device=device)
            # Make class 4 (clothes) win in the center area, class 0 (background) elsewhere
            if num_classes > 4:
                # Background (class 0) is default
                logits[:, 0, :, :] = 5.0
                # Clothing (class 4) wins in center 50% of image
                center_h_start = small_h // 4
                center_h_end = 3 * small_h // 4
                center_w_start = small_w // 4
                center_w_end = 3 * small_w // 4
                logits[
                    :, 4, center_h_start:center_h_end, center_w_start:center_w_end
                ] = 10.0
            return DummySegformerOutput(logits)

    return DummyFeatureExtractor, DummySegformer


def _make_dummy_sam_and_predictor(image_h=10, image_w=10):
    class DummySam:
        def __init__(self, checkpoint=None):
            self.checkpoint = checkpoint

        def to(self, device):
            return self

        def eval(self):
            return self

    class DummySamPredictor:
        def __init__(self, sam):
            self.sam = sam
            self._image = None

        def set_image(self, image_np):
            self._image = image_np

        def predict(
            self, point_coords=None, point_labels=None, box=None, multimask_output=False
        ):
            H, W = self._image.shape[:2]
            if box is None or len(box) == 0:
                return np.zeros((0, H, W), dtype=np.uint8), None, None
            masks = []
            for b in box:
                x0, y0, x1, y1 = b
                x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
                m = np.zeros((H, W), dtype=np.uint8)
                m[y0:y1, x0:x1] = 1
                masks.append(m)
            return np.stack(masks, axis=0), None, None

    return {"vit_h": lambda checkpoint=None: DummySam(checkpoint)}, DummySamPredictor


def test_segment_image_and_dimension(monkeypatch):
    model_path = Path("/models/sam_vit_h_4b8939.pth")
    if not model_path.exists():
        pytest.skip(
            "Skipping segmentation tests because /models/sam_vit_h_4b8939.pth is not present."
        )
    # Get a real test image
    test_images = list(TEST_IMAGE_DIR.glob("*.jpg"))
    assert len(test_images) > 0, "No test images found in fashionpedia_subset"
    img_file = test_images[0]

    # Load image to get its actual dimensions
    test_img = Image.open(img_file)
    img_w, img_h = test_img.size

    DummyFE, DummySeg = _make_dummy_feature_extractor_and_segformer(
        image_h=img_h, image_w=img_w, num_classes=20
    )
    monkeypatch.setattr(segmentation, "SegformerImageProcessor", DummyFE, raising=False)
    monkeypatch.setattr(
        segmentation, "SegformerForSemanticSegmentation", DummySeg, raising=False
    )

    sam_registry, DummySamPredictor = _make_dummy_sam_and_predictor(
        image_h=img_h, image_w=img_w
    )
    monkeypatch.setattr(segmentation, "sam_model_registry", sam_registry, raising=False)
    monkeypatch.setattr(segmentation, "SamPredictor", DummySamPredictor, raising=False)

    # Mock the Hugging Face SAM components to avoid downloading models
    class DummyAutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class DummyAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        segmentation, "AutoProcessor", DummyAutoProcessor, raising=False
    )
    monkeypatch.setattr(
        segmentation, "AutoModelForMaskGeneration", DummyAutoModel, raising=False
    )

    seg = segmentation.Segmentation()

    mask = seg.segment_image(str(img_file))
    assert mask.shape == (img_h, img_w)
    # The real segment_image returns a binary mask (0 or 1), not class labels
    # So we check that we got some positive detections
    assert mask.sum() > 0
    assert mask.max() == 1
    assert mask.min() == 0


def test_refine_with_sam_and_get_full(monkeypatch):
    model_path = Path("/models/sam_vit_h_4b8939.pth")
    if not model_path.exists():
        pytest.skip(
            "Skipping segmentation tests because /models/sam_vit_h_4b8939.pth is not present."
        )
    # Get a real test image
    test_images = list(TEST_IMAGE_DIR.glob("*.jpg"))
    assert len(test_images) > 1, "Need at least 2 test images in fashionpedia_subset"
    img_file = test_images[1]  # Use a different image than the first test

    # Load image to get its actual dimensions
    test_img = Image.open(img_file)
    img_w, img_h = test_img.size

    DummyFE, DummySeg = _make_dummy_feature_extractor_and_segformer(
        image_h=img_h, image_w=img_w, num_classes=20
    )
    monkeypatch.setattr(segmentation, "SegformerImageProcessor", DummyFE, raising=False)
    monkeypatch.setattr(
        segmentation, "SegformerForSemanticSegmentation", DummySeg, raising=False
    )

    sam_registry, DummySamPredictor = _make_dummy_sam_and_predictor(
        image_h=img_h, image_w=img_w
    )
    monkeypatch.setattr(segmentation, "sam_model_registry", sam_registry, raising=False)
    monkeypatch.setattr(segmentation, "SamPredictor", DummySamPredictor, raising=False)

    # Mock the Hugging Face SAM components
    class DummyAutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class DummyAutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        segmentation, "AutoProcessor", DummyAutoProcessor, raising=False
    )
    monkeypatch.setattr(
        segmentation, "AutoModelForMaskGeneration", DummyAutoModel, raising=False
    )

    def fake_get_boxes(self, rough_mask):
        return [[0, 0, 4, 4]]

    monkeypatch.setattr(
        segmentation.Segmentation,
        "_get_bounding_boxes_from_mask",
        fake_get_boxes,
        raising=False,
    )

    seg = segmentation.Segmentation()

    rough = seg.segment_image(str(img_file))
    assert rough.shape == (img_h, img_w)

    img_np = np.array(Image.open(img_file).convert("RGB"))
    refined = seg.refine_with_sam(img_np, rough)
    assert refined.shape == rough.shape
    # Check that refined mask has some positive pixels
    assert refined.sum() > 0

    img_out, refined_out = seg.get_full_segmentation(str(img_file))
    assert isinstance(img_out, np.ndarray)
    assert refined_out.shape == rough.shape
    assert img_out.shape[:2] == (img_h, img_w)
