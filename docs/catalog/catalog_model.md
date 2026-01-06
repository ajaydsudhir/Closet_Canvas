 # Closet Canvas — Catalog README

Overview
- This project extracts clothing regions from images, refines segmentation with SAM, and computes normalized CLIP embeddings for the extracted foreground.
- Key modules:
  - src/catalog/embeddings.py — implementation (prepare models, segment, refine, extract).
  - src/catalog/segmentation.py - implementation (prepare model, compute embeddings).
  - tests/catalog/test_segmentation.py — pytest for testing segmentation.
  - tests/catalog/test_embeddings.py — pytest for testing embeddings.

Model files and defaults
- SegFormer model: default "sayeed99/segformer_b3_clothes" (downloaded via Hugging Face).
- SAM checkpoint: by default downloads from hugging face or else uses local `closet_canvas/models/sam_vit_h_4b8939.pth` (Get the checkpoint from `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`).
- Optional finetuned CLIP weights: by default downloads from hugging face or else uses local path given via prepare_models param (default ../models/finetuned_clip.pt).

Example usecase shown in docs/catalog/catalog_example_use.py
It needs some images in closet_canvas/test_images/fashionpedia_subset

We use this to go to top level for calling the library
'''project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))'''

Running tests
- Tests are in tests/catalog/test_embeddings.py and tests/catalog/test_segmentation.py which include:
  - Py tests for extract_foreground, compute_embedding_from_array, and SAM refinement logic.
  - Calls the library functions
