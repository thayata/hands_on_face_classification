from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# ---------------------------
# Config
# ---------------------------
KNOWN_FACES_DIR = Path("known_faces")
IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
MODEL_DIR = Path.home() / ".cache" / "opencv_face_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
    "face_recognition_sface_2021dec.onnx"
)

UNKNOWN_LABEL = "Unknown"


# ---------------------------
# Utility
# ---------------------------
def ensure_file(path: Path, url: str) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)
    return path


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    rgb = np.asarray(img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(bgr)


def safe_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x / (np.linalg.norm(x) + eps)


@st.cache_resource
def load_models() -> Tuple[cv2.FaceDetectorYN, cv2.FaceRecognizerSF]:
    """Download models if needed and load OpenCV DNN face detector/recognizer."""
    try:
        yunet_model = str(ensure_file(YUNET_PATH, YUNET_URL))
        sface_model = str(ensure_file(SFACE_PATH, SFACE_URL))
    except Exception as e:
        raise RuntimeError(
            "Failed to download required ONNX model files. "
            "Please check your internet connection once, or pre-download the model files "
            f"into {MODEL_DIR}. Original error: {e}"
        )

    try:
        detector = cv2.FaceDetectorYN.create(
            model=yunet_model,
            config="",
            input_size=(320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
        )
        recognizer = cv2.FaceRecognizerSF.create(sface_model, "")
    except Exception as e:
        raise RuntimeError(
            "Your OpenCV build does not include FaceDetectorYN / FaceRecognizerSF. "
            "Install a recent OpenCV package, for example:\n"
            "  pip install -U opencv-python numpy pillow streamlit\n"
            f"Original error: {e}"
        )

    return detector, recognizer


def detect_faces(detector: cv2.FaceDetectorYN, bgr: np.ndarray) -> List[np.ndarray]:
    h, w = bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr)
    if faces is None:
        return []
    # Each row: [x, y, w, h, lmk1x, lmk1y, ..., score]
    faces = sorted(faces, key=lambda f: float(f[-1]), reverse=True)
    return [np.asarray(face, dtype=np.float32) for face in faces]


def face_bbox(face: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = face[:4]
    return int(x), int(y), int(x + w), int(y + h)


def extract_embedding(
    recognizer: cv2.FaceRecognizerSF,
    bgr: np.ndarray,
    face: np.ndarray,
) -> np.ndarray:
    aligned = recognizer.alignCrop(bgr, face)
    feat = recognizer.feature(aligned)
    return safe_normalize(feat)


def build_face_database(
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
    known_dir: Path,
) -> Tuple[np.ndarray, List[str]]:
    embeddings: List[np.ndarray] = []
    labels: List[str] = []

    if not known_dir.exists():
        st.warning(f"Known faces directory not found: {known_dir}")
        return np.empty((0, 128), dtype=np.float32), []

    for person_dir in sorted(known_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name

        image_files: List[Path] = []
        for pattern in IMAGE_EXTS:
            image_files.extend(sorted(person_dir.glob(pattern)))
        if not image_files:
            continue

        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    bgr = preprocess_image(img)
            except Exception as e:
                st.warning(f"Failed to open {img_path}: {e}")
                continue

            faces = detect_faces(detector, bgr)
            if not faces:
                st.warning(f"No face found in {img_path}. Skipping.")
                continue

            # Use the most confident / largest face.
            face = max(faces, key=lambda f: float(f[2] * f[3]))
            try:
                emb = extract_embedding(recognizer, bgr, face)
            except Exception as e:
                st.warning(f"Failed to extract embedding from {img_path}: {e}")
                continue

            embeddings.append(emb)
            labels.append(person_name)

    if not embeddings:
        return np.empty((0, 128), dtype=np.float32), []

    return np.vstack(embeddings).astype(np.float32), labels


@st.cache_resource
def get_face_database() -> Tuple[np.ndarray, List[str]]:
    detector, recognizer = load_models()
    return build_face_database(detector, recognizer, KNOWN_FACES_DIR)


def classify_face(
    face_emb: np.ndarray,
    db_embs: np.ndarray,
    db_labels: List[str],
    threshold: float = 0.35,
) -> Tuple[str, float]:
    if db_embs.shape[0] == 0:
        return "Unknown (empty DB)", 0.0

    sims = db_embs @ safe_normalize(face_emb)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_label = db_labels[best_idx]

    if best_score < threshold:
        return UNKNOWN_LABEL, best_score
    return best_label, best_score


def draw_bbox_and_label(
    bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str,
    score: float,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label} ({score:.2f})"
    cv2.putText(
        bgr,
        text,
        (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return bgr


# ---------------------------
# Streamlit UI
# ---------------------------
def main() -> None:
    st.set_page_config(page_title="Face Classification (OpenCV DNN, Python 3.13)", layout="centered")
    st.title("Face Classification using OpenCV DNN + Streamlit")

    st.markdown(
        """
This app performs **face identity classification** without InsightFace.

It uses:
- **YuNet** for face detection
- **SFace** for face embedding extraction
- **cosine similarity** for identity matching

**How to use:**
1. Put reference images into `known_faces/<person_name>/*.jpg` (or `.png`, `.jpeg`, `.bmp`, `.webp`).
2. Each subfolder name (`<person_name>`) is treated as a **class label**.
3. Run this app and upload an image.

        """
    )

    threshold = st.sidebar.slider(
        "Cosine similarity threshold (higher = stricter, more 'Unknown')",
        min_value=0.10,
        max_value=0.90,
        value=0.6,
        step=0.01,
    )

    st.sidebar.markdown("**Known faces directory:**")
    st.sidebar.code(str(KNOWN_FACES_DIR.resolve()), language="bash")
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "For better results, use several clear, front-facing images per person. "
        "If a known person is classified as `Unknown`, try lowering the threshold a little."
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    try:
        detector, recognizer = load_models()
        db_embs, db_labels = get_face_database()
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.info(
            "Recommended install:\n"
            "`pip install -U streamlit pillow numpy opencv-python`"
        )
        return

    st.caption(f"Loaded {len(db_labels)} reference face image(s).")

    if uploaded_file is None:
        st.info("Please upload an image to start.")
        return

    try:
        with Image.open(uploaded_file) as image:
            image_rgb = image.convert("RGB")
            st.image(image_rgb, caption="Uploaded image", use_container_width=True)
            bgr = preprocess_image(image_rgb)
    except Exception as e:
        st.error(f"Failed to read uploaded image: {e}")
        return

    faces = detect_faces(detector, bgr)
    if not faces:
        st.error("No faces detected in the uploaded image.")
        return

    st.write(f"Detected **{len(faces)}** face(s).")

    annotated = bgr.copy()
    results: List[Tuple[int, str, float, Tuple[int, int, int, int]]] = []

    for i, face in enumerate(faces):
        try:
            face_emb = extract_embedding(recognizer, bgr, face)
            label, score = classify_face(face_emb, db_embs, db_labels, threshold)
            bbox = face_bbox(face)
            results.append((i, label, score, bbox))
            annotated = draw_bbox_and_label(annotated, bbox, label, score)
        except Exception as e:
            st.warning(f"Failed to process face {i + 1}: {e}")

    rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(rgb_annotated, caption="Detected faces & predictions", use_container_width=True)

    st.subheader("Classification results")
    for idx, label, score, bbox in results:
        st.write(
            f"**Face {idx + 1}** — Predicted: `{label}`, "
            f"cosine similarity: `{score:.3f}`, "
            f"bbox: `{list(map(int, bbox))}`"
        )


if __name__ == "__main__":
    main()
