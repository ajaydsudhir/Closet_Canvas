from pathlib import Path
import traceback
import sys
import random
import cv2
import numpy as np

from closet_canvas.catalog.segmentation import Segmentation
from closet_canvas.catalog.embeddings import Embeddings


# ------------------------ Utilities ------------------------ #
def find_test_images(root: Path):
    """
    Find test images in 'data/raw/catalog_test/fashionpedia_subset'.
    """
    specific_path = (
        root / ".." / "data" / "raw" / "catalog_test" / "fashionpedia_subset"
    )
    if specific_path.is_dir():
        return specific_path
    return None


def progress_bar(current_step, total_steps, message="Loading", newline=True):
    bar_length = 20
    fraction = current_step / total_steps
    filled_length = int(bar_length * fraction)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    percent = int(fraction * 100)
    sys.stdout.write(f"\r{message}: \t |{bar}| {percent}%")
    if newline:
        sys.stdout.write("\n")
    sys.stdout.flush()


def save_refined_mask(img_path, refined_mask, mask_type="mask", folder=None):
    """Save binary mask to disk as colored image for visualization."""
    if refined_mask is None or refined_mask.size == 0:
        print(f"No refined mask for {img_path.name}, skipping save.")
        return

    if refined_mask.ndim == 3:
        refined_mask = refined_mask[:, :, 0]

    mask_uint8 = (refined_mask * 255).astype(np.uint8)

    try:
        mask_color = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    except cv2.error:
        mask_color = mask_uint8

    if folder is None:
        folder = Path(__file__).parent
    folder.mkdir(parents=True, exist_ok=True)

    out_path = folder / f"refined_mask_{mask_type}_{img_path.stem}.png"
    cv2.imwrite(str(out_path), mask_color)
    print(f"Refined mask saved to: {out_path}")


def overlay_mask(image, mask, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """Overlay a binary mask on an image."""
    if mask is None or mask.size == 0:
        return image.copy()

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.float32),
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_norm = mask.astype(np.float32)
    if mask_norm.max() > 0:
        mask_norm = (mask_norm / mask_norm.max() * 255).astype(np.uint8)
    else:
        mask_norm = mask_norm.astype(np.uint8)

    mask_colored = cv2.applyColorMap(mask_norm, colormap)
    overlayed = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    return overlayed


# ------------------------ Main Pipeline ------------------------ #
def main(
    project_root: Path,
    max_images: int | None = None,
    randomize: bool = True,
    seed: int | None = None,
):
    test_images_dir = find_test_images(project_root)
    if not test_images_dir:
        print(f"No 'test_images' directory found under {project_root}.")
        return

    seg, emb = None, None

    progress_bar(3, 30, message="Loading segmentation model")
    try:
        seg = Segmentation()
    except Exception:
        print("Failed to initialize Segmentation. Segmentation calls will be skipped.")
        traceback.print_exc()

    progress_bar(6, 30, message="Loading embeddings model")
    try:
        emb = Embeddings()
    except Exception:
        print("Failed to initialize Embeddings. Embedding calls will be skipped.")
        traceback.print_exc()

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [
        p
        for p in test_images_dir.rglob("*")
        if p.suffix.lower() in image_exts and p.is_file()
    ]

    if not images:
        print(f"No images found in {test_images_dir}")
        return

    if max_images is not None and max_images > 0:
        if randomize:
            rng = random.Random(seed)
            images = rng.sample(images, min(max_images, len(images)))
        else:
            images = images[:max_images]

    progress_bar(9, 30, message="Processing images", newline=False)
    segmented_results, segmented_person_results, embedding_results = [], [], []

    for idx, img_path in enumerate(images):
        step = 10 + int(idx * (28 - 10 + 1) / len(images))
        progress_bar(step, 30, message="Processing images", newline=False)

        # Segmentation
        if seg:
            try:
                img_np, refined_mask = seg.get_full_segmentation(str(img_path))
                unique_vals = (
                    sorted(set(refined_mask.flatten().tolist()))
                    if refined_mask is not None
                    else None
                )
                segmented_results.append((img_path, unique_vals))
                # save_refined_mask(img_path, refined_mask, mask_type="cloth")
            except Exception:
                print(f"Segmentation failed for {img_path.name}")
                traceback.print_exc()

        # Person segmentation
        if seg:
            try:
                img_np, refined_mask = seg.get_person_segmentation(str(img_path))
                unique_vals = (
                    sorted(set(refined_mask.flatten().tolist()))
                    if refined_mask is not None
                    else None
                )
                segmented_person_results.append((img_path, unique_vals))
                # save_refined_mask(img_path, refined_mask, mask_type="person")
            except Exception:
                print(f"Person segmentation failed for {img_path.name}")
                traceback.print_exc()

        # Embeddings
        if emb:
            try:
                vec = emb.get_embedding_vector(str(img_path))
                preview = vec[:5].tolist() if vec is not None else None
                embedding_results.append((img_path, preview))
            except Exception:
                print(f"Embedding failed for {img_path.name}")
                traceback.print_exc()

    progress_bar(30, 30, message="Processing complete")

    # print("\nSegmentation Results top 5:")
    # for img_path, unique_vals in segmented_results[:5]:
    #     print(f"  {img_path.name}: unique mask values = {unique_vals}")

    # print("\nPerson Segmentation Results top 5:")
    # for img_path, unique_vals in segmented_person_results[:5]:
    #     print(f"  {img_path.name}: unique mask values = {unique_vals}")

    print("\nEmbedding Results top 5:")
    for img_path, preview in embedding_results[:5]:
        print(f"  {img_path.name}: embedding preview = {preview}")

    if images and seg:
        try:
            img = cv2.imread(str(images[0]))
            if img is not None:
                cloth_mask = seg.get_full_segmentation(str(images[0]))[1]
                person_mask = seg.get_person_segmentation(str(images[0]))[1]

                cloth_overlay = overlay_mask(img, cloth_mask, colormap=cv2.COLORMAP_JET)
                person_overlay = overlay_mask(
                    img, person_mask, colormap=cv2.COLORMAP_OCEAN
                )

                combined = np.hstack([img, cloth_overlay, person_overlay])

                out_path = project_root / f"preview_combined_{images[0].stem}.png"
                cv2.imwrite(str(out_path), combined)
                print(f"Preview saved to {out_path}")
                cv2.imshow("Segmentation Preview", combined)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except Exception:
            print("Failed to generate preview.")
            traceback.print_exc()
    print("Done. L0L")


# ------------------------ Entry Point ------------------------ #
if __name__ == "__main__":
    progress_bar(1, 30, message="Starting up")
    project_root = Path(__file__).resolve().parents[1]
    main(project_root, max_images=2, randomize=True, seed=69)
