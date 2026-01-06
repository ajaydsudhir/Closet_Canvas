import numpy as np
import torch
import cv2
import os
from pathlib import Path
from PIL import Image

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
)

from segment_anything import SamPredictor, sam_model_registry


class Segmentation:
    def __init__(self, segformer_name="sayeed99/segformer_b3_clothes"):
        # Check GPU availability
        cuda_available = torch.cuda.is_available()
        print(f"[Segmentation] CUDA available: {cuda_available}")

        device = torch.device("cuda" if cuda_available else "cpu")
        self.device = device

        # --- Load SegFormer ---
        self.image_processor = SegformerImageProcessor.from_pretrained(segformer_name)
        self.segformer = (
            SegformerForSemanticSegmentation.from_pretrained(segformer_name)
            .eval()
            .to(device)
        )

        # --- Load Meta SAM (segment-anything) ---
        # Use environment variable or fall back to relative path
        model_dir = os.environ.get(
            "MODEL_DIR", str(Path(__file__).parent.parent.parent.parent / "models")
        )
        model_file = os.environ.get("HF_MODEL_FILE", "sam_vit_h_4b8939.pth")
        sam_checkpoint = Path(model_dir) / model_file

        if not sam_checkpoint.exists():
            raise FileNotFoundError(f"SAM model not found at {sam_checkpoint}")

        print(f"[Segmentation] Loading SAM model from {sam_checkpoint}")
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
        sam.to(device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)

        print(f"[Segmentation] Models loaded successfully on {device}")

    def segment_person(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.segformer(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        clothes_mask = np.isin(pred_seg, list(range(1, 17))).astype(np.uint8)
        return clothes_mask

    def segment_image(self, image_path: str):
        """Perform person segmentation on the image using SegFormer."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.segformer(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        clothes_mask = np.isin(pred_seg, [4, 5, 6, 7, 8, 16, 17]).astype(np.uint8)
        return clothes_mask

    def refine_with_sam(self, image, mask):
        self.sam_predictor.set_image(image)

        y, x = np.where(mask > 0)
        bbox = np.array([x.min(), y.min(), x.max(), y.max()])

        masks, scores, logits = self.sam_predictor.predict(box=bbox[None, :])
        refined_mask = masks[np.argmax(scores)]
        return refined_mask

    def sam(self, image_np: np.ndarray, mask: np.ndarray, padding: int = 10):
        """
        Refine rough segmentation mask using Hugging Face SAM base model.
        Returns refined mask as uint8 binary mask.
        """
        processor = self.sam_hf_processor
        model = self.sam_hf_model

        # Find positive pixels
        y, x = np.where(mask > 0)
        if len(x) == 0 or len(y) == 0:
            print("No positive pixels found in mask.")
            return (mask > 0).astype(np.uint8)

        # Compute bbox with padding
        x_min = max(int(x.min()) - padding, 0)
        y_min = max(int(y.min()) - padding, 0)
        x_max = min(int(x.max()) + padding, image_np.shape[1] - 1)
        y_max = min(int(y.max()) + padding, image_np.shape[0] - 1)
        bbox = [[float(x_min), float(y_min), float(x_max), float(y_max)]]

        # Ensure RGB
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)
        if image_np.ndim == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        pil_image = Image.fromarray(image_np)

        # Step 1: get image embeddings
        with torch.no_grad():
            image_inputs = processor(images=pil_image, return_tensors="pt").to(
                self.device
            )
            image_embeddings = model.get_image_embeddings(image_inputs.pixel_values)

        # Step 2: prepare prompt with the box
        inputs = processor(
            images=pil_image,
            input_boxes=[bbox],
            image_embeddings=image_embeddings,
            return_tensors="pt",
        ).to(self.device)

        original_size = pil_image.size[::-1]  # (height, width)
        reshaped_size = inputs.pixel_values.shape[-2:]  # (H, W)

        # Step 3: forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # Step 4: post-process masks
        masks = processor.post_process_masks(
            outputs.pred_masks,
            original_sizes=[original_size],
            reshaped_input_sizes=[reshaped_size],
        )

        refined_mask = masks[0][0].cpu().numpy()
        refined_mask = (refined_mask > 0.5).astype(np.uint8)

        return refined_mask

    def get_full_segmentation(self, image_path: str):
        """Run SegFormer segmentation and refine with SAM 2."""
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        rough_mask = self.segment_image(image_path)
        refined_mask = self.refine_with_sam(image_np, rough_mask)

        return image_np, refined_mask

    def get_person_segmentation(self, image_path: str):
        """Run person segmentation using SegFormer."""
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        rough_mask = self.segment_person(image_path)
        refined_mask = self.refine_with_sam(image_np, rough_mask)
        return image_np, refined_mask
