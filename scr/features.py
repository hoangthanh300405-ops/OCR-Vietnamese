from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.preprocessing import pre_processing


_WIN_SIZE = (128, 256)
_CELL_SIZE = (8, 8)
_BLOCK_SIZE = (8, 8)
_BLOCK_STRIDE = (8, 8)
_NBINS = 27
_DERIV_APERTURE = 1
_WIN_SIGMA = -1
_HIST_NORM_TYPE = 0
_L2_HYS_THRESH = 0.2
_GAMMA_CORR = True
_NLEVELS = 64
_SIGNED_GRADIENT = False

_hog = cv2.HOGDescriptor(
    _WIN_SIZE, _BLOCK_SIZE, _BLOCK_STRIDE,
    _CELL_SIZE, _NBINS, _DERIV_APERTURE,
    _WIN_SIGMA, _HIST_NORM_TYPE, _L2_HYS_THRESH,
    _GAMMA_CORR, _NLEVELS, _SIGNED_GRADIENT,
)

_LBP_CELL_SIZE = (16, 16)
_LBP_N_BINS = 59

_SIFT_SIZE = (128, 256)
_SIFT_GRID_STEP = 16
_SIFT_KEYPOINT_SIZE = 16
_SIFT = cv2.SIFT_create()


def l2_normalize(vector: np.ndarray, axis: int | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector, axis=axis, keepdims=axis is not None)
    return vector / (norm + 1e-8)


def _lbp_uniform_mapping() -> np.ndarray:
    mapping = np.full(256, 58, dtype=np.uint8)
    idx = 0
    for i in range(256):
        bits = (i >> np.arange(8)) & 1
        transitions = np.sum(np.abs(np.diff(np.r_[bits, bits[0]])))
        if transitions <= 2:
            mapping[i] = idx
            idx += 1
    return mapping


_LBP_MAPPING = _lbp_uniform_mapping()


def _compute_lbp(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    padded = np.pad(image, 1, mode="edge")
    center = padded[1:h + 1, 1:w + 1]

    n1 = (padded[1:h + 1, 2:w + 2] >= center).view(np.uint8)
    n2 = (padded[0:h, 2:w + 2] >= center).view(np.uint8)
    n3 = (padded[0:h, 1:w + 1] >= center).view(np.uint8)
    n4 = (padded[0:h, 0:w] >= center).view(np.uint8)
    n5 = (padded[1:h + 1, 0:w] >= center).view(np.uint8)
    n6 = (padded[2:h + 2, 0:w] >= center).view(np.uint8)
    n7 = (padded[2:h + 2, 1:w + 1] >= center).view(np.uint8)
    n8 = (padded[2:h + 2, 2:w + 2] >= center).view(np.uint8)

    lbp = n1 | (n2 << 1) | (n3 << 2) | (n4 << 3) | (n5 << 4) | (n6 << 5) | (n7 << 6) | (n8 << 7)
    return _LBP_MAPPING[lbp]


def _compute_lbp_histogram(lbp_image: np.ndarray) -> np.ndarray:
    h, w = lbp_image.shape
    cell_h, cell_w = _LBP_CELL_SIZE
    n_cells_y = h // cell_h
    n_cells_x = w // cell_w
    histograms = []
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            cell = lbp_image[cy * cell_h:(cy + 1) * cell_h, cx * cell_w:(cx + 1) * cell_w]
            hist = np.bincount(cell.ravel(), minlength=_LBP_N_BINS).astype(np.float32)
            histograms.append(l2_normalize(hist))
    return np.concatenate(histograms).astype(np.float32)


def _prepare_sift_image(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.resize(gray, _SIFT_SIZE, interpolation=cv2.INTER_AREA)


def _dense_keypoints(width: int, height: int) -> list[cv2.KeyPoint]:
    return [
        cv2.KeyPoint(float(x), float(y), _SIFT_KEYPOINT_SIZE)
        for y in range(_SIFT_GRID_STEP // 2, height, _SIFT_GRID_STEP)
        for x in range(_SIFT_GRID_STEP // 2, width, _SIFT_GRID_STEP)
    ]


def compute_dense_sift_descriptors(image: np.ndarray) -> np.ndarray:
    sift_image = _prepare_sift_image(image)
    keypoints = _dense_keypoints(sift_image.shape[1], sift_image.shape[0])
    _, descriptors = _SIFT.compute(sift_image, keypoints)
    if descriptors is None:
        return np.empty((0, 128), dtype=np.float32)
    return descriptors.astype(np.float32)


def fit_sift_codebook(
    image_paths: Iterable[str | Path],
    codebook_size: int = 128,
    max_descriptors: int = 60000,
) -> np.ndarray:
    descriptors = []
    total = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        desc = compute_dense_sift_descriptors(image)
        if desc.size == 0:
            continue
        remaining = max_descriptors - total
        if remaining <= 0:
            break
        if len(desc) > remaining:
            step = max(1, len(desc) // remaining)
            desc = desc[::step][:remaining]
        descriptors.append(desc)
        total += len(desc)

    if not descriptors:
        raise ValueError("Could not extract any SIFT descriptors for codebook training.")

    samples = np.vstack(descriptors).astype(np.float32)
    if len(samples) < codebook_size:
        raise ValueError(f"Need at least {codebook_size} SIFT descriptors, got {len(samples)}.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.01)
    _compactness, _labels, centers = cv2.kmeans(
        samples,
        codebook_size,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    return centers.astype(np.float32)


def _compute_sift_bow(image: np.ndarray, codebook: np.ndarray | None) -> np.ndarray:
    if codebook is None:
        return np.zeros(0, dtype=np.float32)

    descriptors = compute_dense_sift_descriptors(image)
    if descriptors.size == 0:
        return np.zeros(len(codebook), dtype=np.float32)

    # Squared L2 distances to visual words. Dense SIFT count is small, so a
    # direct matrix is simple and fast enough.
    x2 = np.sum(descriptors * descriptors, axis=1, keepdims=True)
    c2 = np.sum(codebook * codebook, axis=1, keepdims=True).T
    distances = x2 + c2 - 2.0 * descriptors @ codebook.T
    assignments = np.argmin(distances, axis=1)

    hist = np.bincount(assignments, minlength=len(codebook)).astype(np.float32)
    hist = np.sqrt(hist)
    return l2_normalize(hist)


def create_raw_vector(file_path: str | Path, codebook: np.ndarray | None = None) -> np.ndarray:
    image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {file_path}")

    processed = pre_processing(image)
    hog_vec = _hog.compute(processed).flatten().astype(np.float32)
    hog_vec = l2_normalize(hog_vec)

    lbp_image = _compute_lbp(processed)
    lbp_vec = _compute_lbp_histogram(lbp_image)
    lbp_vec = l2_normalize(lbp_vec)

    sift_vec = _compute_sift_bow(image, codebook)
    vector = np.concatenate([hog_vec, lbp_vec, sift_vec]).astype(np.float32)
    return l2_normalize(vector)


@dataclass
class PCAModel:
    mean: np.ndarray
    components: np.ndarray

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        reduced = (vectors - self.mean) @ self.components.T
        return reduced.astype(np.float32)


def fit_pca(vectors: np.ndarray, output_dim: int = 512) -> PCAModel:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("PCA input must be a 2D array.")
    n_samples, n_features = vectors.shape
    n_components = min(output_dim, n_samples - 1, n_features)
    if n_components <= 0:
        raise ValueError("Need at least two vectors to fit PCA.")

    mean = vectors.mean(axis=0, keepdims=True).astype(np.float32)
    centered = (vectors - mean).astype(np.float32)

    # Dual PCA is much faster here because n_samples is far smaller than the
    # HOG+LBP+SIFT raw feature dimension.
    gram = (centered @ centered.T) / max(1, n_samples - 1)
    eigenvalues, eigenvectors_small = np.linalg.eigh(gram.astype(np.float32))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order][:n_components]
    eigenvectors_small = eigenvectors_small[:, order][:, :n_components]

    scale = np.sqrt(np.maximum(eigenvalues, 1e-8) * max(1, n_samples - 1))
    components = (eigenvectors_small.T @ centered) / scale[:, None]
    components = l2_normalize(components.astype(np.float32), axis=1)
    return PCAModel(mean=mean, components=components.astype(np.float32))


def save_pickle(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path):
    with Path(path).open("rb") as f:
        return pickle.load(f)


def create_vector(
    file_path: str | Path,
    codebook: np.ndarray | None = None,
    pca_model: PCAModel | None = None,
) -> np.ndarray:
    vector = create_raw_vector(file_path, codebook=codebook)
    if pca_model is not None:
        vector = pca_model.transform(vector)[0]
    return l2_normalize(vector)
