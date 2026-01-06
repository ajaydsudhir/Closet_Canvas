import cv2
import torch
from ultralytics import YOLO
import os
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Segmentation:
    def __init__(self, model_name="yolov8s.pt", conf=0.3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check if model exists in MODEL_DIR, otherwise use default path
        model_dir = os.environ.get("MODEL_DIR", "")
        if model_dir:
            model_path = os.path.join(model_dir, model_name)
            if os.path.exists(model_path):
                model_name = model_path
                logging.info(f"Using YOLO model from MODEL_DIR: {model_path}")

        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.conf = conf
        logging.info(f"YOLO model '{model_name}' loaded on {self.device}")

    def crop_people(self, image, save_dir=None):
        if isinstance(image, np.ndarray):
            img = image
            img_name = "in_memory_image"
        elif isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {image}")
            img_name = os.path.basename(image)
        else:
            raise TypeError("'image' must be a numpy array or valid file path.")

        results = self.model.predict(
            source=img,
            device=self.device,
            conf=self.conf,
            classes=[0],  # class 0 = person
            verbose=False,
        )

        detections = []
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            conf_score = float(results[0].boxes.conf[i])
            crop = img[y1:y2, x1:x2]

            saved_path = None
            if save_dir:
                if crop.shape[2] == 3 and np.mean(crop[:, :, 0] - crop[:, :, 2]) < 0:
                    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

                crop_name = f"{os.path.splitext(img_name)[0]}_person{i}.jpg"
                saved_path = os.path.join(save_dir, crop_name)
                cv2.imwrite(saved_path, crop)

            detections.append(
                {
                    "crop": crop,
                    "bounding_box": [x1, y1, x2, y2],
                    "confidence": conf_score,
                    "saved_path": saved_path,
                }
            )

        if detections:
            logging.info(f"Detected {len(detections)} person(s) in {img_name}")
        else:
            logging.warning(f"No persons detected in {img_name}")

        return detections
