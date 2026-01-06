from huggingface_hub import hf_hub_download
from PIL import Image
import torch

import open_clip
import warnings

warnings.filterwarnings("ignore", message="QuickGELU mismatch*")


class Embeddings:
    def __init__(self):
        clip_arch = "ViT-B/32"
        # local_finetuned_path="closet_canvas/models/finetuned_clip.pt"
        hf_repo = "patrickjohncyh/fashion-clip"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_arch, pretrained="openai", quick_gelu=True
        )
        clip_path = hf_hub_download(repo_id=hf_repo, filename="pytorch_model.bin")

        try:
            state_dict = torch.load(clip_path, map_location=device)
            if "CLIP" in state_dict:
                self.clip_model.load_state_dict(state_dict["CLIP"], strict=False)
            else:
                self.clip_model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(
                f"⚠️  Could not load finetuned weights ({e}); continuing with base weights."
            )

        self.clip_model.eval().to(device)

    def generate_embedding(self, image_path: str):
        """
        Generate a normalized CLIP embedding for the image at image_path.
        Returns a numpy array (1,D).
        """
        device = next(self.clip_model.parameters()).device

        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = self.clip_model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy()

    def get_embedding_dimension(self):
        """
        Returns the dimension of the CLIP embeddings.
        """
        return self.clip_model.visual.output_dim

    def get_embedding_vector(self, image_path: str):
        """
        Returns the embedding vector as a 1D numpy array.
        """
        embedding = self.generate_embedding(image_path)
        return embedding.flatten()
