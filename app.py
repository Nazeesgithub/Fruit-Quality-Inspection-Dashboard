import csv
import base64
import glob
import os
from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from inference import FruitClassifier


st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="wide")

MODEL_CANDIDATES = [
    os.path.join("models", "fruit_mobilenetv2.keras"),
    os.path.join("models", "fruit_efficientnetb0.keras"),
]
LABELS_PATH = os.path.join("models", "labels.json")
UPLOAD_DIR = "uploads"
LOG_DIR = "logs"
HISTORY_CSV = os.path.join(LOG_DIR, "prediction_history.csv")

ALLOWED_EXTS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}
MAX_FILE_MB = 5
SAMPLE_ROOTS = ["Sample Images", os.path.join("datasets", "original_data_set")]

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def ensure_history_file():
    if not os.path.exists(HISTORY_CSV):
        with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "filename",
                "fruit",
                "condition",
                "predicted_class",
                "raw_class",
                "confidence",
                "best_k",
            ])


def read_history() -> pd.DataFrame:
    ensure_history_file()
    hist = pd.read_csv(HISTORY_CSV)
    if "fruit" not in hist.columns or "condition" not in hist.columns:
        derived = hist.get("predicted_class", pd.Series(dtype=str)).fillna("unknown").map(split_prediction_label)
        hist["fruit"] = [item[0] for item in derived]
        hist["condition"] = [item[1] for item in derived]
        ordered_cols = [
            "timestamp",
            "filename",
            "fruit",
            "condition",
            "predicted_class",
            "raw_class",
            "confidence",
            "best_k",
        ]
        existing_cols = [col for col in ordered_cols if col in hist.columns]
        hist = hist[existing_cols]
        hist.to_csv(HISTORY_CSV, index=False)
    return hist


def append_history(row):
    ensure_history_file()
    with open(HISTORY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def build_history_row(source_name: str, result: dict, fruit_label: str, condition_label: str) -> list:
    return [
        datetime.now().isoformat(timespec="seconds"),
        source_name,
        fruit_label,
        condition_label,
        result["predicted_class"],
        result["raw_class"],
        round(result["confidence"], 6),
        result["best_k"],
    ]


def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_sample_images(roots=None) -> Dict[str, List[str]]:
    if roots is None:
        roots = SAMPLE_ROOTS
    if isinstance(roots, str):
        roots = [roots]

    samples: Dict[str, List[str]] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue

        for class_name in sorted(os.listdir(root)):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            paths = []
            for ext in ALLOWED_EXTS:
                paths.extend(glob.glob(os.path.join(class_dir, f"*.{ext}")))
            if paths:
                samples.setdefault(class_name, [])
                samples[class_name].extend(sorted(paths))

    for class_name in list(samples.keys()):
        samples[class_name] = sorted(dict.fromkeys(samples[class_name]))
    return samples


def load_image_from_path(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def image_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def summarize_result(result: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"class": k, "probability": v} for k, v in result["probabilities"].items()]
    ).sort_values("probability", ascending=False)


def split_prediction_label(label: str) -> tuple[str, str]:
    normalized = (label or "").strip().lower().replace("_", " ")
    if not normalized or normalized == "unknown":
        return "Unknown", "Unknown"

    for prefix in ("fresh ", "rotten "):
        if normalized.startswith(prefix):
            condition = prefix.strip().title()
            fruit = normalized[len(prefix):].strip().title()
            return fruit, condition

    if normalized == "mixed":
        return "Mixed", "Unknown"

    return normalized.title(), "Unknown"


def decode_overlay_png_base64(payload: str) -> np.ndarray:
    raw = base64.b64decode(payload)
    data = np.frombuffer(raw, dtype=np.uint8)
    overlay = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if overlay is None:
        raise ValueError("Could not decode overlay image returned from API.")
    return overlay


def run_prediction(clf: FruitClassifier, image_bgr: np.ndarray, threshold: float) -> dict:
    return clf.predict(image_bgr, unknown_threshold=threshold)


def run_prediction_via_api(api_url: str, image_bgr: np.ndarray, threshold: float) -> dict:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("The requests package is required for API mode. Install from requirements.txt.") from exc

    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Could not encode image for API request.")

    files = {"file": ("image.png", buffer.tobytes(), "image/png")}
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        params={"unknown_threshold": threshold},
        files=files,
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    payload["overlay"] = decode_overlay_png_base64(payload.pop("overlay_png_base64"))
    return payload


def check_api_health(api_url: str) -> Tuple[bool, str]:
    try:
        import requests
    except ImportError:
        return False, "The requests package is not installed."

    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return False, str(exc)

    model_loaded = payload.get("model_loaded", False)
    classes = payload.get("classes", [])
    return True, f"status={payload.get('status', 'unknown')} | model_loaded={model_loaded} | classes={classes}"


def validate_upload(file) -> tuple[bool, str]:
    ext = file.name.split(".")[-1].lower()
    if ext not in ALLOWED_EXTS:
        return False, f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTS))}"

    file_size_mb = len(file.getvalue()) / (1024 * 1024)
    if file_size_mb > MAX_FILE_MB:
        return False, f"File too large: {file_size_mb:.2f} MB (max {MAX_FILE_MB} MB)"

    return True, ""


def main():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Outfit:wght@400;600;700&display=swap');

            .stApp {
                background:
                    radial-gradient(circle at 12% 8%, rgba(0, 255, 170, 0.12), transparent 35%),
                    radial-gradient(circle at 90% 15%, rgba(0, 163, 255, 0.10), transparent 32%),
                    linear-gradient(165deg, #06080d 0%, #0a1117 50%, #030507 100%);
                color: #dce7ef;
                font-family: 'Outfit', sans-serif;
            }
            h1, h2, h3, h4, h5, h6 {
                font-family: 'Space Grotesk', sans-serif !important;
                letter-spacing: 0.01em;
            }
            .stApp [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(9,14,20,0.96), rgba(6,10,15,0.96));
                border-right: 1px solid rgba(126, 248, 204, 0.14);
            }
            .hero {
                padding: 1.4rem 1.5rem;
                border-radius: 22px;
                background: linear-gradient(135deg, #09131c 0%, #122634 52%, #0f3a35 100%);
                color: #f3fbff;
                border: 1px solid rgba(126, 248, 204, 0.22);
                box-shadow: 0 20px 48px rgba(0, 0, 0, 0.45);
                animation: fadeSlide 650ms ease-out;
            }
            .hero h1 {
                margin: 0;
                font-size: 2.3rem;
            }
            .hero p {
                margin: 0.4rem 0 0 0;
                opacity: 0.92;
                font-size: 1rem;
            }
            .metric-card {
                padding: 1rem 1rem 0.85rem 1rem;
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(15, 26, 36, 0.96), rgba(10, 19, 28, 0.96));
                border: 1px solid rgba(126, 248, 204, 0.14);
                box-shadow: 0 12px 28px rgba(0, 0, 0, 0.32);
                min-height: 110px;
            }
            .metric-title {
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #8ab8cb;
                margin-bottom: 0.35rem;
            }
            .metric-value {
                font-size: 1.7rem;
                font-weight: 700;
                color: #dcfbff;
                margin-bottom: 0.2rem;
            }
            .metric-subtitle {
                font-size: 0.88rem;
                color: #9ab0be;
            }
            .panel {
                padding: 1rem 1.1rem;
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(14, 23, 32, 0.96), rgba(8, 15, 22, 0.98));
                border: 1px solid rgba(126, 248, 204, 0.12);
                box-shadow: 0 14px 30px rgba(0, 0, 0, 0.35);
                animation: fadeSlide 700ms ease-out;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.4rem;
                background: rgba(9, 16, 24, 0.74);
                border: 1px solid rgba(126, 248, 204, 0.14);
                border-radius: 12px;
                padding: 0.2rem;
            }
            .stTabs [data-baseweb="tab"] {
                background: transparent;
                color: #89a6b5;
                border-radius: 9px;
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(140deg, rgba(0, 255, 170, 0.17), rgba(0, 163, 255, 0.18));
                color: #dcfbff;
            }
            .stButton > button,
            .stDownloadButton > button {
                background: linear-gradient(120deg, #00b88f, #00a0d9);
                border: none;
                color: #04151a;
                font-weight: 700;
            }
            .stButton > button:hover,
            .stDownloadButton > button:hover {
                filter: brightness(1.06);
            }
            .stDataFrame, div[data-testid="stTable"] {
                border: 1px solid rgba(126, 248, 204, 0.15);
                border-radius: 12px;
                overflow: hidden;
            }
            @keyframes fadeSlide {
                from {
                    opacity: 0;
                    transform: translateY(8px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1>Fruit Quality Inspection Dashboard</h1>
            <p>Upload or select an image to run automated fruit detection, segmentation, and class prediction for fast quality-check and labeling workflows.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)

    if model_path is None or not os.path.exists(LABELS_PATH):
        st.warning("Model not found. Train first: python train_classifier.py")
        st.stop()

    clf = FruitClassifier(model_path=model_path, labels_path=LABELS_PATH)
    sample_map = load_sample_images()
    hist = read_history()
    class_labels = [clf.idx_to_label[i] for i in sorted(clf.idx_to_label)]
    class_summary = ", ".join(label.title() for label in class_labels)

    default_api_url = os.environ.get("API_URL", "http://127.0.0.1:8000")
    try:
        default_api_url = st.secrets.get("API_URL", default_api_url)
    except Exception:
        # Local runs often do not have a secrets.toml file.
        pass

    with st.sidebar:
        st.header("Controls")
        unknown_threshold = st.slider("Unknown confidence threshold", min_value=0.3, max_value=0.9, value=0.60, step=0.01)
        use_api_backend = st.toggle("Use API backend", value=False)
        api_url = st.text_input("API URL", value=default_api_url)
        if use_api_backend:
            if st.button("Test API connection", use_container_width=True):
                ok, message = check_api_health(api_url)
                if ok:
                    st.success(f"API connected: {message}")
                else:
                    st.error(f"API health check failed: {message}")
        show_details = st.toggle("Show technical details", value=True)
        st.caption(f"Loaded model: {os.path.basename(model_path)}")
        st.caption("Best for demo use: upload real fruit images taken from different angles and lighting.")
        st.caption("API mode is free: use local FastAPI for local runs or a public FastAPI URL for Streamlit Cloud.")

    st.markdown("### Quick status")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        image_card("Model", os.path.basename(model_path), "CNN transfer learning")
    with s2:
        image_card("Classes", f"{len(class_labels)} + unknown", class_summary)
    with s3:
        image_card("History", str(len(hist)), "saved predictions")
    with s4:
        image_card("Threshold", f"{unknown_threshold:.2f}", "unknown cutoff")

    tab_predict, tab_history, tab_system = st.tabs(["Predict", "History", "System"])

    with tab_predict:
        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            source_mode = st.radio("Choose input source", ["Upload image", "Try a sample image"], horizontal=True)

            image_bgr = None
            source_name = None
            save_name = None

            if source_mode == "Upload image":
                uploaded = st.file_uploader("Upload fruit image", type=list(ALLOWED_EXTS), key="uploader")
                if uploaded is not None:
                    ok, msg = validate_upload(uploaded)
                    if not ok:
                        st.error(msg)
                        st.stop()

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_name = f"{ts}_{uploaded.name}"
                    save_path = os.path.join(UPLOAD_DIR, save_name)

                    with open(save_path, "wb") as f:
                        f.write(uploaded.getbuffer())

                    image_bytes = np.asarray(bytearray(uploaded.getvalue()), dtype=np.uint8)
                    image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                    source_name = uploaded.name
            else:
                if not sample_map:
                    st.warning("No sample images found in Sample Images/")
                else:
                    sample_class = st.selectbox("Sample class", list(sample_map.keys()))
                    sample_file = st.selectbox(
                        "Sample image",
                        sample_map[sample_class],
                        format_func=lambda p: os.path.basename(p),
                    )
                    if sample_file:
                        image_bgr = load_image_from_path(sample_file)
                        source_name = os.path.basename(sample_file)
                        save_name = f"sample_{os.path.basename(sample_file)}"

            run_button = st.button("Run prediction", type="primary", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.subheader("What the system sees")
            if image_bgr is not None:
                st.image(bgr_to_rgb(image_bgr), use_container_width=True)
            else:
                st.info("Choose an image to see the result here.")
            st.markdown("</div>", unsafe_allow_html=True)

        if run_button:
            if image_bgr is None:
                st.warning("Please upload or select a sample image first.")
            else:
                try:
                    with st.spinner("Running segmentation and prediction..."):
                        if use_api_backend:
                            result = run_prediction_via_api(api_url, image_bgr, unknown_threshold)
                        else:
                            result = run_prediction(clf, image_bgr, unknown_threshold)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.stop()

                fruit_label, condition_label = split_prediction_label(result["predicted_class"])
                raw_fruit_label, raw_condition_label = split_prediction_label(result["raw_class"])

                st.markdown("#### Prediction result")
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Fruit", fruit_label)
                p2.metric("Condition", condition_label)
                p3.metric("Confidence", f"{result['confidence']:.3f}")
                p4.metric("Margin", f"{result['margin']:.3f}")

                st.caption(
                    f"Final class: {result['predicted_class']} | Raw class: {result['raw_class']}"
                    if result.get("raw_class")
                    else f"Final class: {result['predicted_class']}"
                )

                vis_left, vis_right = st.columns(2, gap="large")
                with vis_left:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    st.subheader("Segmented overlay")
                    st.image(bgr_to_rgb(result["overlay"]), use_container_width=True)
                    st.caption(f"Selected K for mask extraction: {result['best_k']}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with vis_right:
                    st.markdown('<div class="panel">', unsafe_allow_html=True)
                    st.subheader("Probability distribution")
                    probs_df = summarize_result(result)
                    st.dataframe(probs_df, use_container_width=True, hide_index=True)
                    st.bar_chart(probs_df.set_index("class"))
                    st.markdown('</div>', unsafe_allow_html=True)

                summary_text = (
                    f"Source: {source_name or 'unknown'}\n"
                    f"Fruit: {fruit_label}\n"
                    f"Condition: {condition_label}\n"
                    f"Final class: {result['predicted_class']}\n"
                    f"Raw class: {result['raw_class']}\n"
                    f"Confidence: {result['confidence']:.4f}\n"
                    f"Margin: {result['margin']:.4f}\n"
                    f"Selected K: {result['best_k']}"
                )
                st.download_button(
                    "Download prediction summary",
                    data=summary_text,
                    file_name=f"fruit_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    use_container_width=True,
                )

                append_history(
                    build_history_row(save_name or source_name or "unknown", result, fruit_label, condition_label)
                )

                if show_details:
                    with st.expander("Show technical details"):
                        st.json(
                            {
                                "model_probabilities": result["model_probabilities"],
                                "color_probabilities": result["color_probabilities"],
                                "color_analysis": result["color_analysis"],
                            }
                        )

    with tab_history:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Prediction history")
        hist = read_history()
        if hist.empty:
            st.info("No predictions yet.")
        else:
            display_cols = [col for col in ["timestamp", "filename", "fruit", "condition", "predicted_class", "raw_class", "confidence", "best_k"] if col in hist.columns]
            st.dataframe(hist.sort_values("timestamp", ascending=False)[display_cols], use_container_width=True)
            count_df = (
                hist.assign(result_label=hist["fruit"].fillna("Unknown") + " / " + hist["condition"].fillna("Unknown"))
                ["result_label"]
                .value_counts()
                .rename_axis("class")
                .reset_index(name="count")
            )
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total predictions", len(hist))
            with col_b:
                st.metric("Unique fruit/condition pairs", hist[["fruit", "condition"]].drop_duplicates().shape[0])
            st.subheader("Class count chart")
            st.bar_chart(count_df.set_index("class"))
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_system:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("How the system works")
        st.write("1. The image is uploaded or selected from a sample folder.")
        st.write("2. The system finds the fruit region using K-Means segmentation.")
        st.write("3. A pretrained CNN backbone classifies fruit freshness and type.")
        st.write("4. The app applies confidence and unknown-threshold rules.")
        st.write("5. The prediction is logged into a history CSV.")
        st.write("6. Optional API mode sends the image to a local backend and receives JSON back.")
        st.subheader("Current setup")
        st.write(f"- Model file: {os.path.basename(model_path)}")
        st.write("- Input size: 224x224")
        st.write(f"- Classes: {class_summary}")
        st.write("- Fallback: unknown when confidence is too low")
        st.write("- Backend: Streamlit UI or free local FastAPI service")
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
