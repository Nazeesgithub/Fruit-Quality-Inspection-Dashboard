import base64
import os
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from inference import FruitClassifier


MODEL_CANDIDATES = [
    os.path.join("models", "fruit_mobilenetv2.keras"),
    os.path.join("models", "fruit_efficientnetb0.keras"),
]
LABELS_PATH = os.path.join("models", "labels.json")
MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)


def _load_classifier() -> FruitClassifier:
    if MODEL_PATH is None or not os.path.exists(LABELS_PATH):
        raise FileNotFoundError("Model files were not found. Train the model first.")
    return FruitClassifier(model_path=MODEL_PATH, labels_path=LABELS_PATH)


def _encode_png_base64(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Could not encode overlay image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


app = FastAPI(title="Fruit Classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = _load_classifier()
CLASS_NAMES = [classifier.idx_to_label[i] for i in sorted(classifier.idx_to_label)]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": os.path.basename(MODEL_PATH) if MODEL_PATH else "unknown",
        "classes": CLASS_NAMES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), unknown_threshold: float = 0.60):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    image_array = np.frombuffer(content, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    result = classifier.predict(image_bgr, unknown_threshold=unknown_threshold)

    return {
        "predicted_class": result["predicted_class"],
        "raw_class": result["raw_class"],
        "confidence": result["confidence"],
        "margin": result["margin"],
        "best_k": result["best_k"],
        "probabilities": result["probabilities"],
        "model_probabilities": result["model_probabilities"],
        "color_probabilities": result["color_probabilities"],
        "color_analysis": result["color_analysis"],
        "overlay_png_base64": _encode_png_base64(result["overlay"]),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_service:app", host="127.0.0.1", port=8000, reload=False)