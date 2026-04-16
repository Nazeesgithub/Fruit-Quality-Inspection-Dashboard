import argparse
import json
import os
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from segmentation_utils import preprocess_for_classifier


CLASS_NAMES = [
    "fresh apple",
    "rotten apple",
    "fresh banana",
    "rotten banana",
    "fresh orange",
    "rotten orange",
]
CLASS_ALIASES = {
    "fresh apple": ["fresh apples", "freshapples", "fresh apple"],
    "rotten apple": ["rotten apples", "rottenapples", "rotten apple"],
    "fresh banana": ["fresh bananas", "freshbanana", "fresh banana"],
    "rotten banana": ["rotten bananas", "rottenbanana", "rotten banana"],
    "fresh orange": ["fresh oranges", "freshoranges", "fresh orange"],
    "rotten orange": ["rotten oranges", "rottenoranges", "rotten orange"],
}


def list_images(dataset_roots):
    if isinstance(dataset_roots, str):
        dataset_roots = [dataset_roots]

    image_paths = []
    labels = []
    for dataset_root in dataset_roots:
        for cls in CLASS_NAMES:
            for folder_name in CLASS_ALIASES[cls]:
                cls_dir = os.path.join(dataset_root, folder_name)
                if not os.path.isdir(cls_dir):
                    continue
                for name in os.listdir(cls_dir):
                    if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                        image_paths.append(os.path.join(cls_dir, name))
                        labels.append(cls)
    return image_paths, labels


def prepare_dataset(image_paths, labels, use_segmentation=True, target_size=(224, 224)):
    x_data = []
    y_data = []

    cls_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    for p, lbl in zip(image_paths, labels):
        img = cv2.imread(p)
        if img is None:
            continue

        if use_segmentation:
            prep = preprocess_for_classifier(img, target_size=target_size)
            proc = prep["input_resized_bgr"]
        else:
            proc = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        proc = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x_data.append(proc)
        y_data.append(cls_to_idx[lbl])

    return np.array(x_data), np.array(y_data), cls_to_idx


def build_model(num_classes=4, base="mobilenetv2", input_shape=(224, 224, 3)):
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    if base == "efficientnetb0":
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            name="efficientnetb0_backbone",
        )
    else:
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            name="mobilenetv2_backbone",
        )

    backbone.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = augmentation(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fine_tune_model(model, base_name, num_layers_to_unfreeze=20):
    backbone = model.get_layer(base_name)
    backbone.trainable = True

    if num_layers_to_unfreeze > 0:
        for layer in backbone.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_confusion_matrix(cm, class_names, out_path, title):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_and_save(y_true, y_pred, class_names, out_json, out_cm_png, title):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
    }

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, class_names, out_cm_png, title)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def train_once(
    image_paths,
    labels,
    use_segmentation,
    model_base,
    model_out,
    metrics_out,
    cm_out,
    epochs=12,
    batch_size=8,
):
    x, y, cls_to_idx = prepare_dataset(image_paths, labels, use_segmentation=use_segmentation)
    if len(x) < 8:
        raise ValueError("Not enough valid images for training.")

    idx_to_cls = {v: k for k, v in cls_to_idx.items()}

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    class_counts = np.bincount(y_train, minlength=len(CLASS_NAMES))
    class_weight = {
        idx: float(len(y_train)) / float(len(CLASS_NAMES) * count)
        for idx, count in enumerate(class_counts)
        if count > 0
    }

    model = build_model(num_classes=len(CLASS_NAMES), base=model_base)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    base_layer_name = "mobilenetv2_backbone" if model_base == "mobilenetv2" else "efficientnetb0_backbone"
    model = fine_tune_model(model, base_layer_name, num_layers_to_unfreeze=20)
    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=max(3, epochs // 2),
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    metrics = evaluate_and_save(
        y_test,
        y_pred,
        class_names=CLASS_NAMES,
        out_json=metrics_out,
        out_cm_png=cm_out,
        title="Confusion Matrix" + (" (With Segmentation)" if use_segmentation else " (Without Segmentation)"),
    )

    model.save(model_out)

    labels_path = os.path.join(os.path.dirname(model_out), "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in idx_to_cls.items()}, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train fruit classifier with optional segmentation preprocessing")
    parser.add_argument("--dataset", type=str, default="datasets/original_data_set")
    parser.add_argument("--extra_dataset", type=str, default="", help="Optional second dataset root to combine with --dataset")
    parser.add_argument("--model_base", type=str, default="mobilenetv2", choices=["mobilenetv2", "efficientnetb0"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_segmentation", action="store_true", help="Train directly on resized images instead of segmentation-preprocessed crops")
    parser.add_argument("--compare_no_seg", action="store_true", help="Also train/evaluate a baseline without segmentation")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    dataset_roots = [args.dataset]
    if args.extra_dataset and os.path.isdir(args.extra_dataset):
        dataset_roots.append(args.extra_dataset)

    image_paths, labels = list_images(dataset_roots)
    if not image_paths:
        raise ValueError(f"No images found under dataset roots: {dataset_roots}")

    counts = defaultdict(int)
    for lbl in labels:
        counts[lbl] += 1
    print("Class distribution:", dict(counts))

    use_segmentation = not args.no_segmentation

    print("Training model WITH segmentation preprocessing..." if use_segmentation else "Training model WITHOUT segmentation preprocessing...")
    with_seg_metrics = train_once(
        image_paths=image_paths,
        labels=labels,
        use_segmentation=use_segmentation,
        model_base=args.model_base,
        model_out=os.path.join("models", "fruit_mobilenetv2.keras" if args.model_base == "mobilenetv2" else "fruit_efficientnetb0.keras"),
        metrics_out=os.path.join("artifacts", "metrics_with_segmentation.json" if use_segmentation else "metrics_without_segmentation.json"),
        cm_out=os.path.join("artifacts", "confusion_matrix_with_segmentation.png" if use_segmentation else "confusion_matrix_without_segmentation.png"),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(("With segmentation metrics:" if use_segmentation else "Without segmentation metrics:"), with_seg_metrics)

    if args.compare_no_seg:
        print("Training baseline WITHOUT segmentation preprocessing...")
        no_seg_metrics = train_once(
            image_paths=image_paths,
            labels=labels,
            use_segmentation=False,
            model_base=args.model_base,
            model_out=os.path.join("models", "fruit_baseline_no_seg.keras"),
            metrics_out=os.path.join("artifacts", "metrics_without_segmentation.json"),
            cm_out=os.path.join("artifacts", "confusion_matrix_without_segmentation.png"),
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print("Without segmentation metrics:", no_seg_metrics)


if __name__ == "__main__":
    main()
