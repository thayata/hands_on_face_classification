import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from insightface.app import FaceAnalysis


# ---------------------------
# Config
# ---------------------------

KNOWN_FACES_DIR = Path("known_faces")
EMBEDDING_SIZE = 512  # typical dimension for many InsightFace models


# ---------------------------
# Helper functions
# ---------------------------

@st.cache_resource
def load_insightface_model() -> FaceAnalysis:
    """
    Load InsightFace FaceAnalysis model (cached for Streamlit).
    ctx_id = -1 → CPU, 0 → first GPU.
    """
    app = FaceAnalysis(name="buffalo_l")  # you can change model name if needed
    app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU
    return app


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Convert PIL.Image to a contiguous BGR np.ndarray (uint8, HxWx3).
    """
    img = img.convert("RGB")
    arr = np.array(img)  # RGB
    # Use OpenCV to ensure BGR + contiguous
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr


def build_face_database(
    app: FaceAnalysis,
    known_dir: Path,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a database of embeddings and labels from known_faces/<person>/*.jpg.

    Returns:
        embeddings: np.ndarray of shape (N, d), L2-normalized
        labels: list of length N
    """
    embeddings: List[np.ndarray] = []
    labels: List[str] = []

    if not known_dir.exists():
        st.warning(f"Known faces directory not found: {known_dir}")
        return np.empty((0, EMBEDDING_SIZE), dtype=np.float32), []

    for person_dir in sorted(known_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name

        image_files = (
            list(person_dir.glob("*.jpg"))
            + list(person_dir.glob("*.jpeg"))
            + list(person_dir.glob("*.png"))
        )
        if not image_files:
            continue

        for img_path in image_files:
            try:
                img = Image.open(img_path)
            except Exception as e:
                st.warning(f"Failed to open {img_path}: {e}")
                continue

            bgr = preprocess_image(img)
            faces = app.get(bgr)

            if len(faces) == 0:
                st.warning(f"No face found in {img_path}. Skipping.")
                continue

            # Use the first detected face in this image
            face = faces[0]
            emb = face.normed_embedding  # already L2-normalized in most models
            emb = np.asarray(emb, dtype=np.float32)

            # Extra safety: normalize again
            norm = np.linalg.norm(emb) + 1e-8
            emb = emb / norm

            embeddings.append(emb)
            labels.append(person_name)

    if len(embeddings) == 0:
        return np.empty((0, EMBEDDING_SIZE), dtype=np.float32), []

    embeddings_array = np.vstack(embeddings).astype(np.float32)

    # Normalize all DB embeddings (safety; should be already normalized)
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-8
    embeddings_array = embeddings_array / norms

    return embeddings_array, labels


@st.cache_resource
def get_face_database() -> Tuple[np.ndarray, List[str]]:
    """
    Cached loader for the (embeddings, labels) DB.
    """
    app = load_insightface_model()
    embeddings, labels = build_face_database(app, KNOWN_FACES_DIR)
    return embeddings, labels


def classify_face(
    face_emb: np.ndarray,
    db_embs: np.ndarray,
    db_labels: List[str],
    threshold: float = 0.30,
) -> Tuple[str, float]:
    """
    Compare face_emb with database and return (best_label, best_score).
    Scores are cosine similarity in [~0, 1].
    If best_score < threshold, return ("Unknown", best_score).
    """
    if db_embs.shape[0] == 0:
        return "Unknown (empty DB)", 0.0

    # Ensure face embedding is float32 and normalized
    face_emb = np.asarray(face_emb, dtype=np.float32)
    norm = np.linalg.norm(face_emb) + 1e-8
    face_emb = face_emb / norm

    # db_embs are assumed normalized, so dot product = cosine similarity
    sims = db_embs @ face_emb  # shape (N,)

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    best_label = db_labels[best_idx]

    if best_score < threshold:
        return "Unknown", best_score
    return best_label, best_score


def draw_bbox_and_label(
    bgr: np.ndarray,
    bbox,
    label: str,
    score: float,
) -> np.ndarray:
    """
    Draw bounding box and label text on BGR image.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label} ({score:.2f})"
    cv2.putText(
        bgr,
        text,
        (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return bgr


# ---------------------------
# Streamlit UI
# ---------------------------

def main():
    st.set_page_config(page_title="Face Classification (InsightFace)", layout="centered")
    st.title("Face Classification using InsightFace + Streamlit")

    st.markdown(
        """
This app performs **face identity classification** using InsightFace.

**How to use:**
1. Put reference images into `known_faces/<person_name>/*.jpg` (or `.png`, `.jpeg`).
2. Each subfolder name (`<person_name>`) is treated as a **class label**.
3. Run this app and upload an image – detected faces will be matched to the closest known identity.
        """
    )

    threshold = st.sidebar.slider(
        "Similarity threshold (higher = stricter, more 'Unknown')",
        min_value=0.20,
        max_value=0.80,
        value=0.70,
        step=0.01,
    )

    st.sidebar.markdown("**Known faces directory:**")
    st.sidebar.code(str(KNOWN_FACES_DIR.resolve()), language="bash")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "If a known person is classified as `Unknown`, try **lowering the threshold** "
        "or adding more / clearer reference images."
    )

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
    )

    # Load model and face database (both cached)
    app = load_insightface_model()
    db_embs, db_labels = get_face_database()

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", width='stretch')

        bgr = preprocess_image(image)
        faces = app.get(bgr)

        if len(faces) == 0:
            st.error("No faces detected in the uploaded image.")
            return

        st.write(f"Detected **{len(faces)}** face(s).")

        results = []
        annotated = bgr.copy()

        for i, face in enumerate(faces):
            # Use normalized embedding from InsightFace (float32)
            emb = np.asarray(face.normed_embedding, dtype=np.float32)

            label, score = classify_face(emb, db_embs, db_labels, threshold)
            results.append((i, label, score, face.bbox))
            annotated = draw_bbox_and_label(annotated, face.bbox, label, score)

        # Show annotated image
        rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(rgb_annotated, caption="Detected faces & predictions", width='stretch')

        # Show table of results
        st.subheader("Classification results")
        for idx, label, score, bbox in results:
            st.write(
                f"**Face {idx+1}** — Predicted: `{label}`, "
                f"similarity: `{score:.3f}`, "
                f"bbox: `{[int(x) for x in bbox]}`"
            )
    else:
        st.info("Please upload an image to start.")


if __name__ == "__main__":
    main()
