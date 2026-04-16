# Fruit Segmentation (K-Means)

This short report documents the provided Jupyter notebook `img-seg.ipynb` and how it satisfies the assignment deliverables.

## Assignment checklist

- Load images using OpenCV or PIL

  - The notebook provides `safe_load_image(path, verbose=True)` which attempts to load using OpenCV (`cv2.imread`) and falls back to Pillow, matplotlib.image, and imageio if OpenCV fails. This fulfills the requirement to load images using OpenCV/PIL and handles files OpenCV cannot read.

  - Note about `verbose`: the `safe_load_image` helper defaults to `verbose=True` so that, during development or when debugging problematic files, the notebook will print which loader succeeded (OpenCV, Pillow, matplotlib, or imageio) and any errors encountered. This diagnostic output is helpful when a file appears in the `Sample Images/` folder but fails to read (it shows whether OpenCV failed and which fallback took over).

  - In the processing/demo cells the loader is called with `verbose=False` to keep notebook output concise once the dataset is healthy. If you see unexpected read failures, set `verbose=True` when calling `safe_load_image()` (or temporarily edit the demo cell) to get full loader diagnostics.

- Convert to suitable color space

  - The function `kmeans_segmentation(image_bgr, K=...)` converts input BGR images to Lab (`cv2.COLOR_BGR2LAB`) before clustering. Lab is used for perceptual color grouping.

- Apply K-Means clustering

  - The notebook uses `sklearn.cluster.KMeans` to cluster pixel colors in Lab space. The `K_values` list is configurable; the demo cell now tries K=6..10 (adjustable) to better handle mixed/small fruit cases.
  - The notebook uses `sklearn.cluster.KMeans` to cluster pixel colors in Lab space. The `K_values` list is configurable; the demo cells use K=[2,3,4] by default (this was restored per user preference). You can increase K if you need to separate more color regions or detect very small/multiple fruits.

- Display and compare original and segmented images

  - `show_pair(original_bgr, segmented_bgr, ...)` displays original and segmented images side-by-side using matplotlib.

- Explain choice of K and observed result

  - The notebook contains a markdown section "Choosing K and observations" describing why small K (2-4) may work for foreground/background and why larger K may help separate nearby fruits or textured regions. The student should add a short paragraph in the notebook (or below) stating which K they chose per image and why — some examples are provided.

- Save segmented outputs

  - Segmented outputs and masks are saved to `segmented_outputs/<class>/` by the demo cell. Masks are saved as PNGs.

- Google Drive downloader and auto-sort (optional)
  - The notebook includes helpers `download_and_sort_folder()` and `download_and_prepare_links()` to download images from Google Drive (requires `gdown`) and place them under `Sample Images/<class>/`.

## What I changed

- Replaced remaining direct `cv2.imread(...)` calls in the processing/demo cell with `safe_load_image(...)` to avoid failures when OpenCV cannot read certain files.
- Increased K search range in the demo cell to 6..10 by default to improve detection of multiple small fruits in mixed images.
- Restored the default K search range in the demo cell to 2..4 (per user request) and ensured all demo reads use `safe_load_image(...)` so image loading is robust when OpenCV alone fails.

## How to run

1. Install dependencies (if not already installed):

```powershell
pip install opencv-python scikit-learn matplotlib numpy pillow imageio scipy gdown
```

2. Populate `Sample Images/` with per-class subfolders: `apple`, `orange`, `banana`, `mixed`. You may use the included Drive downloader cells to fetch and sort images.

3. Run the notebook cells in order. The demo cell will process all images found and write outputs to `segmented_outputs/`.

Note: The demo cells use K values [2,3,4] by default; to try a different set of K values, edit the `K_values` list in the demo cell before running.

## Limitations and remaining suggestions

- Some downloaded files may be corrupted or in formats that trigger OpenCV read failures; `safe_load_image` mitigates this but if all loaders fail you should re-download the file from Drive or replace it.
- The heuristic cluster selection (`select_clusters_by_class`) uses simple HSV thresholds and morphological cleanup. It works well for many cases but may need per-dataset tuning. For robust multi-object segmentation consider a trained segmentation model (Mask R-CNN or a lightweight U-Net) or adding an object detector to crop and process regions.
- If mixed images still miss small fruits, try increasing K further (12–20) and relax morphological area thresholds.
- If mixed images still miss small fruits, try increasing K further (e.g., 8–20) and relax morphological area thresholds. The default K is conservative (2–4) to separate foreground/background with minimal over-segmentation.

## Minimal deliverables checklist (for submission)

- Notebook: `img-seg.ipynb` (contains code, diagnostics, and demo).
- Input images: place them under `Sample Images/<class>/` (not included here). (user must ensure images available)
- Segmented outputs: will be saved under `segmented_outputs/<class>/` after running the demo cell. (generated by notebook)
- Short report: this file `FruitSegmentation_report.md`
