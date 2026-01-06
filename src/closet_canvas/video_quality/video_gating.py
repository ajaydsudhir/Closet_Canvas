import cv2
import os
import gc
from closet_canvas.video_quality.metrics import Metrics as VideoQualityMetrics
from dotenv import load_dotenv


class VideoGating:
    @staticmethod
    def extract_frames(video_path, out_dir, fps_sampling=2):
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / fps_sampling))

        frame_id = 0
        saved_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % frame_interval == 0:
                cv2.imwrite(f"{out_dir}/frame_{saved_id:05d}.jpg", frame)
                saved_id += 1

            frame_id += 1

        cap.release()
        return saved_id

    def evaluate_score(self, frames_dir, threshold=60):
        total = 0
        good = 0

        # Reuse single Metrics instance to avoid loading models multiple times
        try:
            load_dotenv()
        except Exception:
            pass
        use_body_capture = os.getenv("USE_BODY_CAPTURE", "false").lower() in (
            "true",
            "yes",
        )
        vqm = VideoQualityMetrics(use_body_capture=use_body_capture)
        masks = []

        for f in sorted(os.listdir(frames_dir)):
            frame_path = os.path.join(frames_dir, f)
            if not os.path.isfile(frame_path):
                continue
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            score, refined_mask = vqm.score_frame(frame)
            print(
                f"[VideoGating] Frame {f}: score={score}, threshold={threshold}, passed={score >= threshold}"
            )
            if score >= threshold:
                good += 1
                masks.append(refined_mask)
            total += 1

            # Clear frame from memory immediately
            del frame
            # Delete the frame file immediately after processing
            try:
                os.remove(frame_path)
            except OSError:
                pass

        # Clean up metrics and force garbage collection
        del vqm
        gc.collect()

        if total == 0:
            print("[VideoGating] No frames extracted")
            return False, []

        pass_rate = good / total
        passed = pass_rate > 0.70
        print(
            f"[VideoGating] Total frames: {total}, Good frames: {good}, Pass rate: {pass_rate:.2f}, Passed: {passed}"
        )
        return passed, masks

    def evaluate_video_with_segformer_sam(self, video_path, threshold=60):
        temp_dir = "temp_frames"
        n_frames = self.extract_frames(video_path, temp_dir, fps_sampling=2)
        if n_frames == 0:
            return False

        try:
            result, masks = self.evaluate_score(temp_dir, threshold=threshold)
        finally:
            # Clean up temp directory (frames already deleted during processing)
            if os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    pass
            gc.collect()

        return result, masks
