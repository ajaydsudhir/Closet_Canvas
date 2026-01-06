import os
import shutil
from huggingface_hub import hf_hub_download

DEST_DIR = os.environ.get("MODEL_DIR", "/models")
os.makedirs(DEST_DIR, exist_ok=True)

# Determine which model to download based on USE_BODY_CAPTURE flag
USE_BODY_CAPTURE = os.environ.get("USE_BODY_CAPTURE", "false").lower() == "true"

if USE_BODY_CAPTURE:
    # Download YOLO model for body capture
    MODEL_REPO = "Ultralytics/YOLOv8"
    MODEL_FILENAME = "yolov8s.pt"
    MODEL_TYPE = "YOLO"
else:
    # Download SAM model for catalog segmentation
    MODEL_REPO = os.environ.get("HF_MODEL_REPO", "abishek0002/sam")
    MODEL_FILENAME = os.environ.get("HF_MODEL_FILE", "sam_vit_h_4b8939.pth")
    MODEL_TYPE = "SAM"

DEST_PATH = os.path.join(DEST_DIR, MODEL_FILENAME)

if os.path.exists(DEST_PATH):
    print(f"[Model] Already cached: {DEST_PATH}")
else:
    token = os.environ.get("HF_ACCESS_TOKEN")
    if not token and not USE_BODY_CAPTURE:
        raise SystemExit(
            "HF_ACCESS_TOKEN environment variable is required for SAM model"
        )

    print(
        f"[Model] Downloading {MODEL_TYPE} model: {MODEL_FILENAME} from {MODEL_REPO}..."
    )
    try:
        # Download to HF cache, then copy to our MODEL_DIR
        download_kwargs = {
            "repo_id": MODEL_REPO,
            "filename": MODEL_FILENAME,
            "repo_type": "model",
        }
        if token:
            download_kwargs["token"] = token

        downloaded_path = hf_hub_download(**download_kwargs)
        print(f"[Model] Downloaded to HF cache: {downloaded_path}")

        # Copy to our persistent location
        shutil.copy2(downloaded_path, DEST_PATH)
        print(f"[Model] Copied to {DEST_PATH}")
    except Exception as e:
        print(f"[Model] Failed to download: {e}")
        raise SystemExit(1)

print(f"[Model] {MODEL_TYPE} model ready at {DEST_PATH}")
