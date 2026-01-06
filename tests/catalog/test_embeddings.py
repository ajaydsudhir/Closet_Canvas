from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from closet_canvas.catalog import embeddings as embedding


# Path to test images
TEST_IMAGE_DIR = (
    Path(__file__).parent.parent.parent
    / "data"
    / "raw"
    / "catalog_test"
    / "fashionpedia_subset"
)


def _make_dummy_model():
    """
    Return a function compatible with open_clip.create_model_and_transforms
    that yields a tiny dummy model and a simple preprocess function.
    """

    class DummyVisual:
        output_dim = 2

    class DummyModel:
        def __init__(self):
            self.visual = DummyVisual()
            # parameter to provide a .device via next(self.parameters()).device
            self._param = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            return iter([self._param])

        def encode_image(self, x):
            # Return a fixed 1x2 tensor; normalization occurs in the code under test.
            return torch.tensor([[3.0, 4.0]], device=self._param.device)

        def load_state_dict(self, *args, **kwargs):
            # no-op for tests
            return

        def eval(self):
            return self

        def to(self, device):
            self._param.data = self._param.data.to(device)
            return self

    def create_model_and_transforms(*args, **kwargs):
        # preprocess should be a callable function that returns a tensor
        def preprocess(image):
            # Return a dummy tensor with the expected shape
            return torch.zeros(3, 224, 224)

        return DummyModel(), None, preprocess

    return create_model_and_transforms


def test_embeddings(monkeypatch):
    # Monkeypatch the open_clip module used inside the imported embeddings module
    dummy_factory = _make_dummy_model()
    monkeypatch.setattr(
        embedding,
        "open_clip",
        SimpleNamespace(create_model_and_transforms=dummy_factory),
        raising=False,
    )

    # Instantiate the Embeddings class (will use the monkeypatched factory)
    emb = embedding.Embeddings()

    # Use a real fashion image from the test dataset
    test_images = list(TEST_IMAGE_DIR.glob("*.jpg"))
    assert len(test_images) > 0, "No test images found in fashionpedia_subset"
    img_file = test_images[0]

    # Generate embedding and validate shape and normalization
    vec = emb.generate_embedding(str(img_file))
    assert vec.shape == (1, 2)

    # expected normalized vector from [3,4] -> [0.6, 0.8]
    expected = np.array([[3.0, 4.0]])
    expected_norm = expected / np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(vec, expected_norm)

    # dimension and flattened vector checks
    assert emb.get_embedding_dimension() == 2
    flat = emb.get_embedding_vector(str(img_file))
    assert flat.shape == (2,)
    assert np.allclose(flat, expected_norm.flatten())
