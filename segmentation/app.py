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

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features import create_vector, load_pickle, l2_normalize

_MODEL_DIR = _ROOT / "models"


def normalize_background(gray: np.ndarray) -> np.ndarray:
    bg = cv2.medianBlur(gray, 51)
    bg[bg < 10] = 10
    norm = cv2.divide(gray.astype(np.float32), bg.astype(np.float32), scale=255.0)
    return np.clip(norm, 0, 255).astype(np.uint8)


def preprocess_for_segmentation(gray: np.ndarray) -> np.ndarray:
    gray = normalize_background(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # For photographed paper, adaptive thresholding tends to promote paper
    # texture to foreground. Global dark-ink thresholding after background
    # normalization is much cleaner for segmentation.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove ruled paper / underline strokes before word dilation.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, gray.shape[1] // 8), 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
    binary = cv2.subtract(binary, horizontal)

    clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel, iterations=1)

    num_labels, cc, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 20:
            filtered[cc == i] = 255
    return filtered


def _merge_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def _dbscan(points: np.ndarray, eps: float, min_samples: int = 1) -> np.ndarray:
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    if n == 0:
        return labels

    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(distances[i] <= eps)[0].tolist()
        if len(neighbors) < min_samples:
            labels[i] = cluster_id
            cluster_id += 1
            continue

        labels[i] = cluster_id
        queue = list(neighbors)
        while queue:
            j = queue.pop(0)
            if not visited[j]:
                visited[j] = True
                j_neighbors = np.where(distances[j] <= eps)[0].tolist()
                if len(j_neighbors) >= min_samples:
                    for candidate in j_neighbors:
                        if candidate not in queue:
                            queue.append(candidate)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1

    return labels


def segment_words(binary: np.ndarray, debug: bool = False) -> list[tuple[int, int, int, int]]:
    num_labels, cc, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    heights = [
        stats[i, cv2.CC_STAT_HEIGHT]
        for i in range(1, num_labels)
        if stats[i, cv2.CC_STAT_AREA] >= 30
    ]
    median_h = float(np.median(heights)) if heights else max(16.0, binary.shape[0] / 40.0)

    # Dilation joins letters and their diacritics into word-sized blobs. Horizontal
    # size is deliberately moderate; DBSCAN below handles fragmented leftovers.
    kx = max(12, int(median_h * 1.15))
    ky = max(3, int(median_h * 0.25))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if w < 12 or h < 10 or area < 180:
            continue
        if w / max(h, 1) > 20:
            continue
        boxes.append((x, y, w, h))

    if debug:
        cv2.imwrite("debug_dilated.png", dilated)

    if not boxes:
        return []

    centers = np.array([[x + w / 2.0, y + h / 2.0] for x, y, w, h in boxes], dtype=np.float32)
    # Scale x down before DBSCAN so nearby fragments on the same baseline merge,
    # while different lines remain separated by y distance.
    scaled = centers.copy()
    scaled[:, 0] /= max(1.0, median_h * 1.8)
    scaled[:, 1] /= max(1.0, median_h * 0.8)
    labels = _dbscan(scaled, eps=1.35, min_samples=1)

    merged: list[tuple[int, int, int, int]] = []
    for label in sorted(set(labels)):
        group = [boxes[i] for i in np.where(labels == label)[0]]
        merged.append(_merge_boxes(group))

    H, W = binary.shape[:2]
    padded = []
    for x, y, w, h in merged:
        px = max(5, int(w * 0.04))
        py = max(6, int(h * 0.12))
        x0 = max(0, x - px)
        y0 = max(0, y - py)
        x1 = min(W, x + w + px)
        y1 = min(H, y + h + py)
        if (x1 - x0) >= 12 and (y1 - y0) >= 10:
            padded.append((x0, y0, x1 - x0, y1 - y0))

    return sort_reading_order(padded)


def sort_reading_order(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    median_h = float(np.median([h for _, _, _, h in boxes]))
    line_h = max(20.0, median_h * 1.3)
    return sorted(boxes, key=lambda b: (int((b[1] + b[3] / 2) // line_h), b[0]))


def crop_word(gray: np.ndarray, bbox: tuple[int, int, int, int], pad_ratio: float) -> np.ndarray:
    H, W = gray.shape[:2]
    x, y, w, h = bbox
    px = max(8, int(h * pad_ratio))
    py = max(8, int(h * pad_ratio * 0.8))
    x0 = max(0, x - px)
    y0 = max(0, y - py)
    x1 = min(W, x + w + px)
    y1 = min(H, y + h + py)
    crop = gray[y0:y1, x0:x1]
    return cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)


def vote(distances, indices, labels):
    counter = Counter()
    for idx, distance in zip(indices, distances):
        counter[str(labels[idx])] += float(distance)
    return counter.most_common(5)


def segment_and_recognize(
    image_path,
    index_path=_MODEL_DIR / "vector_database.index",
    labels_path=_MODEL_DIR / "word_labels.npy",
    names_path=_MODEL_DIR / "file_names.npy",
    codebook_path=_MODEL_DIR / "codebook.pkl",
    pca_path=_MODEL_DIR / "pca_model.pkl",
    k=7,
    pad_ratio=0.35,
    min_score=0.0,
    debug_crops=False,
):
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
            cv2.putText(debug_img, str(i), (x, max(18, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite("debug_words.png", debug_img)

    index = faiss.read_index(str(index_path))
    labels = np.load(str(labels_path), allow_pickle=True)
    _names = np.load(str(names_path), allow_pickle=True)
    codebook = load_pickle(codebook_path)
    pca_model = load_pickle(pca_path)

    recognized = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i, bbox in enumerate(boxes):
            crop = crop_word(gray, bbox, pad_ratio=pad_ratio)
            crop_path = tmpdir / f"word_{i:03d}.png"
            cv2.imwrite(str(crop_path), crop)

            if debug_crops:
                shutil.copy2(crop_path, Path("debug_crops") / f"word_{i:03d}.png")

            vector = create_vector(crop_path, codebook=codebook, pca_model=pca_model)
            query = l2_normalize(np.asarray([vector], dtype=np.float32), axis=1).astype(np.float32)
            if query.shape[1] != index.d:
                raise ValueError(f"Query dim {query.shape[1]} does not match index dim {index.d}. Rebuild the index.")

            distances, indices = index.search(query, k)
            top = vote(distances[0], indices[0], labels)
            best_label, best_score = top[0]
            top_str = " | ".join(f"{lbl}({score:.3f})" for lbl, score in top[:3])
            print(f"  [word {i:03d}] '{best_label}' score={best_score:.3f} top3: {top_str}")
            if best_score >= min_score:
                recognized.append(best_label)

    return " ".join(recognized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vietnamese OCR using HOG+LBP+dense-SIFT PCA retrieval")
    parser.add_argument("--image", required=True, help="Path to input line/paragraph image")
    parser.add_argument("--index", default=str(_MODEL_DIR / "vector_database.index"))
    parser.add_argument("--labels", default=str(_MODEL_DIR / "word_labels.npy"))
    parser.add_argument("--names", default=str(_MODEL_DIR / "file_names.npy"))
    parser.add_argument("--codebook", default=str(_MODEL_DIR / "codebook.pkl"))
    parser.add_argument("--pca", default=str(_MODEL_DIR / "pca_model.pkl"))
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--pad-ratio", type=float, default=0.35)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--debug-crops", action="store_true")
    args = parser.parse_args()

    text = segment_and_recognize(
        image_path=args.image,
        index_path=args.index,
        labels_path=args.labels,
        names_path=args.names,
        codebook_path=args.codebook,
        pca_path=args.pca,
        k=args.k,
        pad_ratio=args.pad_ratio,
        min_score=args.min_score,
        debug_crops=args.debug_crops,
    )

    print("\nRecognized text:")
    print(text)
