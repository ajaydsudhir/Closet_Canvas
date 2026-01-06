import os
import pytest
import numpy as np
from unittest.mock import patch
import logging
from closet_canvas.recommendation import catalog_recommender

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@pytest.fixture(scope="module")
def recommender():
    return catalog_recommender.CatalogRecommender()


@pytest.fixture(scope="module")
def test_image_folder():
    return "data/raw/catalog_test/fashionpedia_subset"


def get_image_length(folder_path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    return len(
        [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and os.path.splitext(f)[1].lower() in exts
        ]
    )


def test_initialize_clip_model(recommender):
    assert recommender.model is not None
    assert recommender.processor is not None
    assert recommender.device in ["cuda", "cpu"]


def test_compute_image_embeddings(recommender, test_image_folder):
    if not os.path.exists(test_image_folder):
        pytest.skip(f"Test image folder '{test_image_folder}' does not exist.")

    if get_image_length(test_image_folder) < 10:
        pytest.skip(
            f"Test requires at least 10 images, found {get_image_length(test_image_folder)}."
        )

    embeddings = recommender.compute_image_embeddings(test_image_folder, max_images=5)
    assert isinstance(embeddings, dict)
    assert len(embeddings) > 0


def test_compute_text_embedding(recommender):
    text = "A red dress"
    embedding = recommender.compute_text_embedding(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[1] == 512


def test_find_top_n_matches(recommender, test_image_folder):
    if not os.path.exists(test_image_folder):
        pytest.skip(f"Test image folder '{test_image_folder}' does not exist.")

    if get_image_length(test_image_folder) < 10:
        pytest.skip(
            f"Test requires at least 10 images, found {get_image_length(test_image_folder)}."
        )

    embeddings = recommender.compute_image_embeddings(test_image_folder, max_images=5)
    logging.info("Embeddings: %s", embeddings)
    text = "A red dress"
    top_matches = recommender.find_top_n_matches(text, embeddings, top_n=3)
    logging.info("Top Matches: %s", top_matches)
    assert isinstance(top_matches, list)
    assert len(top_matches) == 3
    for match in top_matches:
        assert isinstance(match, tuple)
        assert isinstance(match[0], str)
        assert isinstance(match[1], float)


def test_load_and_precompute_embeddings(recommender, test_image_folder):
    if not os.path.exists(test_image_folder):
        pytest.skip(f"Test image folder '{test_image_folder}' does not exist.")

    if get_image_length(test_image_folder) < 10:
        pytest.skip(
            f"Test requires at least 10 images, found {get_image_length(test_image_folder)}."
        )

    with patch(
        "closet_canvas.recommendation.catalog_recommender.display"
    ) as mock_display:
        embeddings = recommender.load_and_precompute_embeddings(
            test_image_folder, max_images=5, user_query="A red dress", top_n=3
        )
        assert isinstance(embeddings, dict)
        assert len(embeddings) > 0

        assert mock_display.call_count == 3
        for call in mock_display.call_args_list:
            args, kwargs = call
            assert len(args) == 1
            assert isinstance(args[0], str)
            assert os.path.exists(args[0])
