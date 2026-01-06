import cv2
import numpy as np
import tempfile
import os
import gc
from closet_canvas.catalog.segmentation import Segmentation
from closet_canvas.body_capture.segmentation import Segmentation as Segmen

# Singleton instance to avoid reloading heavy models
_SEGMENTATION_INSTANCE = None
_BODY_SEGMENTATION_INSTANCE = None


def get_segmentation():
    global _SEGMENTATION_INSTANCE
    if _SEGMENTATION_INSTANCE is None:
        print("[Metrics] Initializing Segmentation models (one-time)...")
        _SEGMENTATION_INSTANCE = Segmentation()
    return _SEGMENTATION_INSTANCE


def get_segmentation_body_capture():
    global _BODY_SEGMENTATION_INSTANCE
    if _BODY_SEGMENTATION_INSTANCE is None:
        print("[Metrics] Initializing body-capture Segmentation (one-time)...")
        _BODY_SEGMENTATION_INSTANCE = Segmen()
    return _BODY_SEGMENTATION_INSTANCE


class Metrics:
    def __init__(self, use_body_capture=False):
        self.use_body_capture = use_body_capture
        if use_body_capture:
            self.segmentation = get_segmentation_body_capture()
        else:
            self.segmentation = get_segmentation()

    def measure_lighting(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def measure_contrast(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    def measure_sharpness(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def visibility_score(self, frame, box):
        h, w, _ = frame.shape
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        frame_area = w * h
        return area / frame_area

    def person_visibility_score(self, mask):
        total_pixels = mask.size
        person_pixels = int(mask.sum())
        return person_pixels / total_pixels

    def body_cutoff_score(self, box, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box

        cutoff = x1 < 5 or y1 < 5 or x2 > w - 5 or y2 > h - 5
        return 0 if cutoff else 1

    def score_frame(self, frame):
        # Save frame temporarily for segmentation
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, frame)

        try:
            refined_mask = None
            # Get coarse mask using segmentation
            if self.use_body_capture:
                detections = self.segmentation.crop_people(cv2.imread(tmp_path))
                if not detections:
                    return 0, None
                # choose best detection by confidence (or first)
                best = max(detections, key=lambda d: d.get("confidence", 0.0))
                x1, y1, x2, y2 = best["bounding_box"]
                h, w = frame.shape[:2]
                refined_mask = np.zeros((h, w), dtype=np.uint8)
                # clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                refined_mask[y1:y2, x1:x2] = 1
            else:
                # Use get_full_person_segmentation for better results
                _, refined_mask = self.segmentation.get_person_segmentation(tmp_path)
                if refined_mask is None or refined_mask.sum() == 0:
                    return 0, None

            # Get bounding box from mask
            y, x = np.where(refined_mask > 0)
            if len(x) == 0 or len(y) == 0:
                return 0, None
            box = [int(x.min()), int(y.min()), int(x.max()), int(y.max())]

            vis = self.person_visibility_score(refined_mask)
            body_ok = self.body_cutoff_score(box, frame)

            lighting = self.measure_lighting(frame)
            sharpness = self.measure_sharpness(frame)
            contrast = self.measure_contrast(frame)

            score = 0
            score_details = []
            if vis > 0.10:
                score += 40
                score_details.append(f"vis={vis:.2f}(+40)")
            else:
                score_details.append(f"vis={vis:.2f}(+0)")
            if body_ok:
                score += 15
                score_details.append("body_ok(+15)")
            else:
                score_details.append("body_cutoff(+0)")
            if lighting > 60:
                score += 15
                score_details.append(f"light={lighting:.0f}(+15)")
            else:
                score_details.append(f"light={lighting:.0f}(+0)")
            if sharpness > 10:
                score += 20
                score_details.append(f"sharp={sharpness:.0f}(+20)")
            else:
                score_details.append(f"sharp={sharpness:.0f}(+0)")
            if contrast > 20:
                score += 10
                score_details.append(f"contrast={contrast:.0f}(+10)")
            else:
                score_details.append(f"contrast={contrast:.0f}(+0)")

            print(f"[Metrics] Score breakdown: {' | '.join(score_details)} = {score}")

            mask_cv = refined_mask.astype(np.uint8) * 255
            masked = cv2.bitwise_and(frame, frame, mask=mask_cv)

            x1, y1, x2, y2 = box
            cropped = masked[y1:y2, x1:x2]

            # Clean up intermediate variables (only those that exist)
            for v in ("coarse_mask", "frame_rgb"):
                if v in locals():
                    del locals()[v]
            del y, x, box
            gc.collect()

            return score, cropped
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
