import cv2
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return (labels == largest).astype(np.uint8)


def _score_cluster(center_hsv: np.ndarray) -> float:
    h, s, v = center_hsv
    if v < 40:
        return -1.0
    return float(0.65 * s + 0.35 * v)


def find_best_mask(image_bgr: np.ndarray, k_values=(2, 3, 4, 5)):
    img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    h, w = img_lab.shape[:2]
    x = img_lab.reshape((-1, 3)).astype(np.float32)

    best = {
        "score": -1.0,
        "mask": None,
        "k": None,
        "segmented_bgr": None,
    }

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(x)
        centers_lab = km.cluster_centers_.astype(np.uint8)

        segmented_lab = centers_lab[labels].reshape((h, w, 3)).astype(np.uint8)
        segmented_bgr = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)

        centers_bgr = cv2.cvtColor(centers_lab.reshape((1, -1, 3)), cv2.COLOR_LAB2BGR).reshape((-1, 3))
        centers_hsv = cv2.cvtColor(centers_bgr.reshape((1, -1, 3)), cv2.COLOR_BGR2HSV).reshape((-1, 3))

        cluster_scores = np.array([_score_cluster(c) for c in centers_hsv])
        selected_idx = int(np.argmax(cluster_scores))

        mask = (labels.reshape((h, w)) == selected_idx).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        mask = keep_largest_component(mask)

        area_ratio = float(mask.sum()) / float(h * w)
        plausibility = area_ratio if 0.02 <= area_ratio <= 0.9 else area_ratio * 0.5
        total_score = float(cluster_scores[selected_idx]) * plausibility

        if total_score > best["score"]:
            best["score"] = total_score
            best["mask"] = mask
            best["k"] = k
            best["segmented_bgr"] = segmented_bgr

    return best["mask"], best["segmented_bgr"], best["k"]


def crop_with_mask(image_bgr: np.ndarray, mask: np.ndarray, margin_ratio: float = 0.08) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image_bgr

    h, w = image_bgr.shape[:2]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    bw = x_max - x_min + 1
    bh = y_max - y_min + 1
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    x1 = max(0, x_min - mx)
    y1 = max(0, y_min - my)
    x2 = min(w, x_max + mx + 1)
    y2 = min(h, y_max + my + 1)

    return image_bgr[y1:y2, x1:x2]


def preprocess_for_classifier(image_bgr: np.ndarray, target_size=(224, 224)):
    mask, segmented_bgr, best_k = find_best_mask(image_bgr)
    if mask is None:
        mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

    cropped = crop_with_mask(image_bgr, mask)
    full_resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

    overlay = image_bgr.copy()
    overlay[mask == 1] = (
        0.5 * overlay[mask == 1] + 0.5 * np.array([0, 255, 0], dtype=np.uint8)
    ).astype(np.uint8)

    return {
        "input_resized_bgr": resized,
        "full_resized_bgr": full_resized,
        "mask": mask,
        "overlay": overlay,
        "best_k": best_k,
        "segmented_bgr": segmented_bgr,
    }
