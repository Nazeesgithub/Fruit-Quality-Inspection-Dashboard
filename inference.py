import json
import os
from typing import Dict, Tuple

import cv2
import numpy as np

from segmentation_utils import preprocess_for_classifier


DEFAULT_MODEL_PATH = os.path.join("models", "fruit_mobilenetv2.keras")
DEFAULT_LABELS_PATH = os.path.join("models", "labels.json")

DEFAULT_UNKNOWN_THRESHOLD = 0.50
DEFAULT_MARGIN_THRESHOLD = 0.05


class FruitClassifier:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, labels_path: str = DEFAULT_LABELS_PATH):
        try:
            import tensorflow as tf
        except Exception as exc:
            raise ImportError(
                "TensorFlow is required to load the local model.\n"
                "If you are running the Streamlit frontend, either enable the API backend mode or install TensorFlow in your environment."
            ) from exc

        self.model = tf.keras.models.load_model(model_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            self.idx_to_label = json.load(f)
        self.idx_to_label = {int(k): str(v).strip() for k, v in self.idx_to_label.items()}
        self.labels = [self.idx_to_label[i] for i in sorted(self.idx_to_label)]
        normalized = {self._normalize_label(label): label for label in self.labels}
        self.has_rotten_labels = any("rotten" in key for key in normalized)

    @staticmethod
    def _normalize_label(label: str) -> str:
        return label.lower().replace("_", "").replace(" ", "")

    def _label_for(self, candidates):
        normalized = {self._normalize_label(label): label for label in self.labels}
        for candidate in candidates:
            match = normalized.get(self._normalize_label(candidate))
            if match is not None:
                return match
        return None

    def _fresh_rotten_pair(self, label: str):
        key = self._normalize_label(label)
        if key.startswith("fresh"):
            target = "rotten" + key[len("fresh"):]
        elif key.startswith("rotten"):
            target = "fresh" + key[len("rotten"):]
        else:
            return None
        for existing in self.labels:
            if self._normalize_label(existing) == target:
                return existing
        return None

    @staticmethod
    def _hue_distance(h1: float, h2: float) -> float:
        d = abs(float(h1) - float(h2))
        return min(d, 180.0 - d)

    def _color_probabilities(self, image_bgr: np.ndarray, mask: np.ndarray) -> Dict:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        if mask is None or mask.sum() < 100:
            pix = hsv.reshape((-1, 3))
            area_ratio = 1.0
        else:
            pix = hsv[mask > 0]
            area_ratio = float(mask.sum()) / float(mask.shape[0] * mask.shape[1])

        if len(pix) == 0:
            if self.has_rotten_labels:
                base = {label: 1.0 / len(self.labels) for label in self.labels}
            else:
                base = {"apple": 0.25, "banana": 0.25, "orange": 0.25, "mixed": 0.25}
            return base, {"dominant_hue": 0.0, "mean_s": 0.0, "mean_v": 0.0, "hue_std": 0.0, "area_ratio": area_ratio}

        h = pix[:, 0].astype(np.float32)
        s = pix[:, 1].astype(np.float32)
        v = pix[:, 2].astype(np.float32)

        # Emphasize colorful bright pixels when estimating dominant hue.
        w = (s / 255.0) * (v / 255.0) + 1e-6
        hist = np.bincount(h.astype(np.int32), weights=w, minlength=180)
        dominant_h = float(np.argmax(hist))
        hue_std = float(np.sqrt(np.average((h - h.mean()) ** 2, weights=w)))
        mean_s = float(np.average(s, weights=w))
        mean_v = float(np.average(v, weights=w))

        apple_score = 0.0
        banana_score = 0.0
        orange_score = 0.0
        mixed_score = 0.0

        # Apple can be red (near 0/179) or green (roughly 45..85).
        apple_score += max(0.0, 1.0 - self._hue_distance(dominant_h, 0.0) / 20.0)
        apple_score += max(0.0, 1.0 - self._hue_distance(dominant_h, 170.0) / 20.0)
        apple_score += 1.0 if 45.0 <= dominant_h <= 85.0 else 0.0

        banana_score += max(0.0, 1.0 - self._hue_distance(dominant_h, 24.0) / 18.0)
        orange_score += max(0.0, 1.0 - self._hue_distance(dominant_h, 13.0) / 14.0)

        # Mixed tends to have wider hue spread and larger color diversity.
        if hue_std > 26.0:
            mixed_score += min(1.5, (hue_std - 26.0) / 20.0)
        if mean_s > 95 and 0.08 <= area_ratio <= 0.85:
            mixed_score += 0.25

        # Down-weight all color-based confidence if region is weakly saturated.
        saturation_factor = np.clip(mean_s / 120.0, 0.35, 1.0)
        apple_score *= saturation_factor
        banana_score *= saturation_factor
        orange_score *= saturation_factor

        raw = np.array([
            max(apple_score, 0.01),
            max(banana_score, 0.01),
            max(orange_score, 0.01),
            max(mixed_score, 0.01),
        ], dtype=np.float32)
        raw = raw / raw.sum()

        color_probs = {
            "apple": float(raw[0]),
            "banana": float(raw[1]),
            "orange": float(raw[2]),
            "mixed": float(raw[3]),
        }
        color_analysis = {
            "dominant_hue": dominant_h,
            "mean_s": mean_s,
            "mean_v": mean_v,
            "hue_std": hue_std,
            "area_ratio": area_ratio,
            "freshness_score": float(np.clip((mean_s / 150.0) * 0.55 + (mean_v / 180.0) * 0.45, 0.0, 1.0)),
        }
        if not self.has_rotten_labels:
            return color_probs, color_analysis

        freshness_score = color_analysis["freshness_score"]
        fruit_scores = {
            "apple": max(0.05, apple_score),
            "banana": max(0.05, banana_score),
            "orange": max(0.05, orange_score),
        }
        rotten_score = 1.0 - freshness_score

        fresh_labels = {
            "apple": self._label_for(["fresh apple", "freshapple"]),
            "banana": self._label_for(["fresh banana", "freshbanana"]),
            "orange": self._label_for(["fresh orange", "freshoranges"]),
        }
        rotten_labels = {
            "apple": self._label_for(["rotten apple", "rottenapples"]),
            "banana": self._label_for(["rotten banana", "rottenbanana"]),
            "orange": self._label_for(["rotten orange", "rottenoranges"]),
        }

        fresh_rotten_probs = {}
        for fruit, fruit_score in fruit_scores.items():
            fresh_label = fresh_labels[fruit]
            rotten_label = rotten_labels[fruit]
            if fresh_label is None or rotten_label is None:
                continue
            fresh_rotten_probs[fresh_label] = float(fruit_score * freshness_score)
            fresh_rotten_probs[rotten_label] = float(fruit_score * rotten_score)

        total = sum(fresh_rotten_probs.values()) + 1e-8
        for key in fresh_rotten_probs:
            fresh_rotten_probs[key] /= total

        return fresh_rotten_probs, color_analysis

    def predict(self, image_bgr: np.ndarray, unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD) -> Dict:
        prep = preprocess_for_classifier(image_bgr, target_size=(224, 224))
        crop_img = prep["input_resized_bgr"]
        full_img = prep["full_resized_bgr"]

        crop_x = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        full_x = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        crop_x = np.expand_dims(crop_x, axis=0)
        full_x = np.expand_dims(full_x, axis=0)

        crop_probs = self.model.predict(crop_x, verbose=0)[0]
        full_probs = self.model.predict(full_x, verbose=0)[0]
        model_probs_arr = 0.75 * crop_probs + 0.25 * full_probs

        model_probs = {self.idx_to_label[i]: float(p) for i, p in enumerate(model_probs_arr)}
        color_probs, color_analysis = self._color_probabilities(image_bgr, prep["mask"])

        # Combine learned model with hue-based class evidence from the notebook-style segmentation region.
        combined_probs = {}
        for label in self.labels:
            combined_probs[label] = 0.82 * model_probs.get(label, 0.0) + 0.18 * color_probs.get(label, 0.0)

        if self.has_rotten_labels:
            class_bias = {}
            for label in self.labels:
                if "rotten" in self._normalize_label(label):
                    class_bias[label] = 1.00
                else:
                    class_bias[label] = 1.02
            for label, bias in class_bias.items():
                combined_probs[label] *= bias
        else:
            # Calibration to counter frequent mixed-overprediction on this small dataset.
            class_bias = {
                "apple": 1.10,
                "banana": 1.05,
                "orange": 1.06,
                "mixed": 0.82,
            }
            for label, bias in class_bias.items():
                combined_probs[label] *= bias

        # Normalize in case labels order differs.
        total = sum(combined_probs.values()) + 1e-8
        for k in combined_probs:
            combined_probs[k] /= total

        probs = np.array([combined_probs[self.idx_to_label[i]] for i in range(len(self.idx_to_label))], dtype=np.float32)

        ranked = sorted(combined_probs.items(), key=lambda kv: kv[1], reverse=True)
        pred_label, top1 = ranked[0]
        second_label, top2 = ranked[1]
        confidence = float(top1)
        margin = float(top1 - top2)

        def margin_for(label: str) -> float:
            base = combined_probs.get(label, 0.0)
            rival = max((v for k, v in combined_probs.items() if k != label), default=0.0)
            return float(base - rival)

        if not self.has_rotten_labels:
            # If mixed is only slightly above a specific fruit, trust the fruit class.
            if pred_label == "mixed" and second_label in ("apple", "banana", "orange") and margin < 0.16:
                pred_label, confidence = second_label, float(top2)
                margin = float(top2 - ranked[2][1]) if len(ranked) > 2 else float(top2)

            # Guard against frequent apple->mixed mistakes when mixed is only slightly higher.
            if pred_label == "mixed":
                if (combined_probs.get("mixed", 0.0) - combined_probs.get("apple", 0.0)) < 0.10:
                    if color_analysis["dominant_hue"] <= 18 or color_analysis["dominant_hue"] >= 160 or (45 <= color_analysis["dominant_hue"] <= 85):
                        pred_label = "apple"
                        confidence = float(combined_probs["apple"])
                        margin = float(combined_probs["apple"] - combined_probs.get("mixed", 0.0))

            # If apple is low-margin while mixed is still strong, prefer unknown over wrong apple.
            if pred_label == "apple":
                if combined_probs.get("mixed", 0.0) >= 0.40 and (margin < 0.12 or confidence < 0.55):
                    pred_label = "unknown"

            # Class-specific acceptance thresholds: mixed should be harder to accept.
            class_thresholds = {
                "apple": max(0.50, unknown_threshold),
                "banana": max(0.50, unknown_threshold),
                "orange": max(0.50, unknown_threshold),
                "mixed": max(0.64, unknown_threshold + 0.08),
            }
        else:
            # Freshness-aware correction to reduce rotten->fresh mistakes.
            freshness = float(color_analysis.get("freshness_score", 0.5))
            pair_label = self._fresh_rotten_pair(pred_label)
            if pair_label is not None:
                pred_key = self._normalize_label(pred_label)
                if pred_key.startswith("fresh") and freshness < 0.40:
                    if combined_probs.get(pair_label, 0.0) >= confidence - 0.12:
                        pred_label = pair_label
                        confidence = float(combined_probs[pair_label])
                        margin = margin_for(pred_label)
                elif pred_key.startswith("rotten") and freshness > 0.74:
                    if combined_probs.get(pair_label, 0.0) >= confidence - 0.12:
                        pred_label = pair_label
                        confidence = float(combined_probs[pair_label])
                        margin = margin_for(pred_label)

            class_thresholds = {label: max(0.48, unknown_threshold) for label in self.labels}

        margin_threshold = DEFAULT_MARGIN_THRESHOLD
        if pred_label in ("mixed",) or (self.has_rotten_labels and "rotten" in self._normalize_label(pred_label)):
            margin_threshold = max(margin_threshold, 0.11)

        if pred_label == "unknown":
            final_label = "unknown"
        else:
            threshold = class_thresholds.get(pred_label, max(0.50, unknown_threshold))
            final_label = pred_label if (confidence >= threshold and margin >= margin_threshold) else "unknown"

        return {
            "predicted_class": final_label,
            "raw_class": pred_label,
            "confidence": confidence,
            "margin": margin,
            "probabilities": {self.idx_to_label[i]: float(p) for i, p in enumerate(probs)},
            "model_probabilities": model_probs,
            "color_probabilities": color_probs,
            "color_analysis": color_analysis,
            "mask": prep["mask"],
            "overlay": prep["overlay"],
            "best_k": prep["best_k"],
        }


def load_image_from_path(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def predict_from_path(
    image_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    labels_path: str = DEFAULT_LABELS_PATH,
    unknown_threshold: float = DEFAULT_UNKNOWN_THRESHOLD,
) -> Tuple[Dict, np.ndarray]:
    clf = FruitClassifier(model_path=model_path, labels_path=labels_path)
    image_bgr = load_image_from_path(image_path)
    result = clf.predict(image_bgr, unknown_threshold=unknown_threshold)
    return result, image_bgr
