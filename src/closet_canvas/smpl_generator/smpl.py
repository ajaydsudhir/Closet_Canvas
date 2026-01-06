import cv2
import asyncio
import numpy as np
from pathlib import Path
from math import pi, sqrt
from typing import List, Union

import mediapipe as mp


class BodyMeasurementEstimatorAsync:
    def __init__(
        self, model_complexity=2, min_detection_confidence=0.5, max_concurrency=4
    ):
        """
        Async-ready estimator.

        max_concurrency controls how many images run simultaneously.
        """
        mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )
        self.mp_pose = mp_pose

        self.semaphore = asyncio.Semaphore(max_concurrency)

    @staticmethod
    def ellipse_perim(a, b):
        a, b = max(a, b) / 2, min(a, b) / 2
        h = ((a - b) ** 2) / ((a + b) ** 2)
        return pi * (a + b) * (1 + 3 * h / (10 + sqrt(4 - 3 * h)))

    async def _async_pose(self, img):
        """
        Run mediapipe.pose in a background thread asynchronously.
        """
        return await asyncio.to_thread(
            lambda: self.pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        )

    async def _extract_features_from_image(self, img, index):
        """
        Extracts width, depth, height data from a single image.
        Includes detailed feedback if landmarks are partially missing.
        """
        async with self.semaphore:
            results = await self._async_pose(img)

        if not results.pose_landmarks:
            return {
                "index": index,
                "success": False,
                "feedback": "Pose not detected in image",
            }

        Pose = self.mp_pose.PoseLandmark
        lm = results.pose_landmarks.landmark
        h, w = img.shape[:2]

        # HEIGHT PX
        try:
            nose_y = lm[Pose.NOSE].y * h
            ankle_y = max(lm[Pose.LEFT_ANKLE].y, lm[Pose.RIGHT_ANKLE].y) * h
            height_px = ankle_y - nose_y
        except Exception as e:
            print(f"Error computing height: {e}")
            return {
                "index": index,
                "success": False,
                "feedback": "Height landmarks missing (nose or ankles)",
            }

        # WIDTH (left/right required)
        shoulder_feedback = ""
        hip_feedback = ""

        try:
            sh_w = abs(lm[Pose.LEFT_SHOULDER].x - lm[Pose.RIGHT_SHOULDER].x) * w
            shoulder_valid = True
        except Exception as e:
            print(f"Error computing shoulder width: {e}")
            sh_w = None
            shoulder_valid = False
            shoulder_feedback = (
                "Only one shoulder available — cannot compute shoulder width"
            )

        try:
            hip_w = abs(lm[Pose.LEFT_HIP].x - lm[Pose.RIGHT_HIP].x) * w
            hip_valid = True
        except Exception as e:
            print(f"Error computing hip width: {e}")
            hip_w = None
            hip_valid = False
            hip_feedback = "Only one hip landmark available — cannot compute hip width"

        # DEPTH (nose reference)
        try:
            nose_x = lm[Pose.NOSE].x * w
            shoulder_x = max(lm[Pose.LEFT_SHOULDER].x, lm[Pose.RIGHT_SHOULDER].x) * w
            elbow_x = max(lm[Pose.LEFT_ELBOW].x, lm[Pose.RIGHT_ELBOW].x) * w
            hip_x = lm[Pose.LEFT_HIP].x * w

            chest_depth = max(abs(shoulder_x - nose_x), abs(elbow_x - nose_x)) * 0.9
            waist_depth = abs(hip_x - nose_x) * 1.1
            hip_depth = abs(hip_x - nose_x) * 1.15
        except Exception as e:
            print(f"Error computing depths: {e}")
            chest_depth = waist_depth = hip_depth = None

        return {
            "index": index,
            "success": True,
            "height_px": height_px,
            "shoulder_width_px": sh_w,
            "hip_width_px": hip_w,
            "chest_width_px": sh_w * 0.92 if sh_w else None,
            "waist_width_px": hip_w * 0.88 if hip_w else None,
            "chest_depth_px": chest_depth,
            "waist_depth_px": waist_depth,
            "hip_depth_px": hip_depth,
            "feedback": {
                "shoulder_valid": shoulder_valid,
                "shoulder_feedback": shoulder_feedback,
                "hip_valid": hip_valid,
                "hip_feedback": hip_feedback,
            },
        }

    async def estimate(
        self, images: List[Union[str, Path, np.ndarray]], user_height_cm: float
    ):
        """
        Async multi-image inference.
        Supports:
            - file paths
            - numpy images
        Returns fused measurements and per-image feedback.
        """

        # Load images
        loaded = []
        for img in images:
            if isinstance(img, (str, Path)):
                im = cv2.imread(str(img))
                if im is not None:
                    loaded.append(im)
            else:
                loaded.append(img)

        if len(loaded) == 0:
            return {"error": "No valid images provided"}

        # Process all images asynchronously
        tasks = [
            self._extract_features_from_image(img, i) for i, img in enumerate(loaded)
        ]
        results = await asyncio.gather(*tasks)

        # Filter successes
        valid = [r for r in results if r["success"]]

        if len(valid) == 0:
            return {"error": "Pose failed on all images", "details": results}

        # Extract data
        px_heights = [r["height_px"] for r in valid]

        def collect(key):
            return [r[key] for r in valid if r[key] is not None]

        shoulder_w_px = collect("shoulder_width_px")
        chest_w_px = collect("chest_width_px")
        waist_w_px = collect("waist_width_px")
        hip_w_px = collect("hip_width_px")

        chest_d_px = collect("chest_depth_px")
        waist_d_px = collect("waist_depth_px")
        hip_d_px = collect("hip_depth_px")

        # Scale px → cm
        scale = user_height_cm / np.mean(px_heights)

        def avg_scaled(lst):
            return (np.mean(lst) * scale) if lst else 0

        shoulder = avg_scaled(shoulder_w_px)
        chest_w = avg_scaled(chest_w_px)
        waist_w = avg_scaled(waist_w_px)
        hip_w = avg_scaled(hip_w_px)

        chest_d = avg_scaled(chest_d_px)
        waist_d = avg_scaled(waist_d_px)
        hip_d = avg_scaled(hip_d_px)

        # Circumference
        chest_circ = self.ellipse_perim(chest_w, chest_d)
        waist_circ = self.ellipse_perim(waist_w, waist_d)
        hip_circ = self.ellipse_perim(hip_w, hip_d)

        return {
            "height_cm": user_height_cm,
            "shoulder_width_cm": shoulder,
            "chest_circumference_cm": chest_circ,
            "waist_circumference_cm": waist_circ,
            "hip_circumference_cm": hip_circ,
            "waist_to_hip_ratio": waist_circ / hip_circ if hip_circ > 0 else None,
            "images_used": len(valid),
            # Detailed feedback for each image
            "image_feedback": [
                {
                    "index": r["index"],
                    "success": r["success"],
                    "feedback": r.get("feedback", {}),
                }
                for r in results
            ],
        }
