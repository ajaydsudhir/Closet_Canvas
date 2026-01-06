import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from IPython.display import display
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CatalogRecommender:
    def __init__(self):
        self.model, self.processor, self.device = self.initialize_clip_model()

    def initialize_clip_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", use_fast=True
        )
        return model, processor, device

    def compute_image_embeddings(self, image_folder, max_images=10000):
        image_embeddings = {}
        count = 0
        for filename in os.listdir(image_folder):
            if count >= max_images:
                break
            if filename.lower().endswith((".jpg", ".png")):
                image_path = os.path.join(image_folder, filename)
                try:
                    image = Image.open(image_path)
                    inputs = self.processor(
                        images=image, return_tensors="pt", padding=True
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self.model.get_image_features(**inputs)
                    image_embeddings[filename] = outputs.cpu().numpy()
                    count += 1
                except Exception as e:
                    logging.error(f"Error processing {filename}: {e}")
        return image_embeddings

    def compute_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(
            self.device
        )
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy()

    def find_top_n_matches(self, text, image_embeddings, top_n=5):
        text_embedding = self.compute_text_embedding(text)
        image_names = list(image_embeddings.keys())
        image_features = np.vstack(list(image_embeddings.values()))
        similarities = cosine_similarity(text_embedding, image_features).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        return [(image_names[i], float(similarities[i])) for i in top_indices]

    def load_and_precompute_embeddings(
        self, image_folder, max_images=1000, user_query=None, top_n=5
    ):
        logging.info(f"Checking for images in folder: {image_folder}")
        if not os.path.exists(image_folder):
            logging.error(f"Error: Folder '{image_folder}' does not exist.")
            return None
        image_files = [
            f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))
        ]
        if not image_files:
            logging.error(f"Error: No image files found in folder '{image_folder}'.")
            return None
        logging.info("Computing image embeddings...")
        image_embeddings = self.compute_image_embeddings(
            image_folder, max_images=max_images
        )
        logging.info("Image embeddings computed.")

        if user_query:
            logging.info(f"Finding top-{top_n} matches for query: '{user_query}'")
            top_matches = self.find_top_n_matches(user_query, image_embeddings, top_n)
            for match in top_matches:
                logging.info(f"Image: {match[0]}, Similarity: {match[1]:.4f}")
                display(os.path.join(image_folder, match[0]))

        return image_embeddings
