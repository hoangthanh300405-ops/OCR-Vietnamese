from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.cnn_features import EMBED_DIM, WordEmbeddingNet, create_cnn_vector

# ── re-use unchanged segmentation helpers from existing segmentation/app.py ──
from segmentation.app import (
    preprocess_for_segmentation,
    segment_words,
    crop_word,
)

_MODEL_DIR = _ROOT / "models"


def _load_model(model_path: Path, device: str) -> WordEmbeddingNet:
    checkpoint = torch.load(model_path, map_location=device)
    embed_dim = checkpoint.get("embed_dim", EMBED_DIM)
    model = WordEmbeddingNet(embed_dim=embed_dim, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def vote(distances: np.ndarray, indices: np.ndarray, labels: np.ndarray):
    """Weighted vote — same logic as original segmentation/app.py."""
    counter: Counter = Counter()
    for idx, dist in zip(indices, distances):
        counter[str(labels[idx])] += float(dist)
    return counter.most_common(5)


def cnn_segment_and_recognize(
    image_path,
    index_path=_MODEL_DIR / "cnn_vector_database.index",
    labels_path=_MODEL_DIR / "cnn_word_labels.npy",
    names_path=_MODEL_DIR / "cnn_file_names.npy",
    model_path=_MODEL_DIR / "cnn_embedding.pt",
    k: int = 7,
    pad_ratio: float = 0.35,
    min_score: float = 0.0,
    debug_crops: bool = False,
    device: str | None = None,
) -> str:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = preprocess_for_segmentation(gray)

    if debug_crops:
        cv2.imwrite("debug_binary.png", binary)
        debug_dir = Path("debug_crops")
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        debug_dir.mkdir(exist_ok=True)

    boxes = segment_words(binary, debug=debug_crops)
    if not boxes:
        print("No words detected.")
        return ""

    if debug_crops:
        debug_img = img.copy()
        for i, (x, y, w, h) in enumerate(boxes):
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug_img, str(i), (x, max(18, y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite("debug_words.png", debug_img)

    # ── Load FAISS + model ──────────────────────────────────────────────────
    index = faiss.read_index(str(index_path))
    # IndexHNSWFlat: tune efSearch for speed vs recall tradeoff at query time.
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = 64

    labels = np.load(str(labels_path), allow_pickle=True)
    model = _load_model(Path(model_path), device)
    print(f"Using CNN model on {device}. Index contains {index.ntotal} vectors.")

    recognized: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i, bbox in enumerate(boxes):
            crop = crop_word(gray, bbox, pad_ratio=pad_ratio)
            crop_path = tmpdir / f"word_{i:03d}.png"
            cv2.imwrite(str(crop_path), crop)

            if debug_crops:
                shutil.copy2(crop_path, Path("debug_crops") / f"word_{i:03d}.png")

            vector = create_cnn_vector(crop_path, model, device=device)
            query = vector[np.newaxis, :].astype(np.float32)

            if query.shape[1] != index.d:
                raise ValueError(
                    f"Query dim {query.shape[1]} ≠ index dim {index.d}. "
                    "Rebuild index with cnn_database.py."
                )

            # HNSW returns L2 distances; convert to similarity score for
            # display consistency (lower L2 = higher cosine for unit vectors).
            distances, indices = index.search(query, k)
            # Invert L2 so vote() still rewards small distance with high weight.
            sim_scores = 2.0 - np.clip(distances[0], 0, 4.0)
            top = vote(sim_scores, indices[0], labels)
            best_label, best_score = top[0]
            top_str = " | ".join(f"{lbl}({s:.3f})" for lbl, s in top[:3])
            print(f"  [word {i:03d}] '{best_label}' score={best_score:.3f}  top3: {top_str}")

            if best_score >= min_score:
                recognized.append(best_label)

    return " ".join(recognized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vietnamese OCR – CNN ResNet18 embedding + FAISS HNSW retrieval"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--index", default=str(_MODEL_DIR / "cnn_vector_database.index"))
    parser.add_argument("--labels", default=str(_MODEL_DIR / "cnn_word_labels.npy"))
    parser.add_argument("--names", default=str(_MODEL_DIR / "cnn_file_names.npy"))
    parser.add_argument("--model", default=str(_MODEL_DIR / "cnn_embedding.pt"))
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--pad-ratio", type=float, default=0.35)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--debug-crops", action="store_true")
    parser.add_argument("--device", default=None, help="cuda | cpu (auto-detect if omitted)")
    args = parser.parse_args()

    text = cnn_segment_and_recognize(
        image_path=args.image,
        index_path=args.index,
        labels_path=args.labels,
        names_path=args.names,
        model_path=args.model,
        k=args.k,
        pad_ratio=args.pad_ratio,
        min_score=args.min_score,
        debug_crops=args.debug_crops,
        device=args.device,
    )
    print("\nRecognized text:")
    print(text)
