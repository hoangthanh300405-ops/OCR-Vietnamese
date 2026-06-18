from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.features import (
    create_raw_vector,
    fit_pca,
    fit_sift_codebook,
    l2_normalize,
    save_pickle,
)


def load_labels(csv_path: str | Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 3:
                labels[row[1]] = row[2]
    return labels


def _word_files(image_dir: str | Path) -> list[str]:
    return sorted(f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))


def _extract_batch(paths: list[Path], codebook: np.ndarray) -> np.ndarray:
    vectors = [create_raw_vector(path, codebook=codebook) for path in paths]
    return np.asarray(vectors, dtype=np.float32)


def build_vector_database(
    image_dir: str | Path = "./Vietnam/train_word",
    csv_path: str | Path = "./Vietnam/train_word.csv",
    output_index: str | Path = "./models/vector_database.index",
    output_labels: str | Path = "./models/word_labels.npy",
    output_names: str | Path = "./models/file_names.npy",
    output_codebook: str | Path = "./models/codebook.pkl",
    output_pca: str | Path = "./models/pca_model.pkl",
    pca_dim: int = 512,
    pca_samples: int = 5000,
    sift_words: int = 128,
    sift_train_images: int = 1500,
    sift_max_descriptors: int = 60000,
    batch_size: int = 500,
    max_images: int | None = None,
):
    image_dir = Path(image_dir)
    files = _word_files(image_dir)
    if max_images is not None:
        files = files[:max_images]
    labels_dict = load_labels(csv_path)
    if not files:
        raise FileNotFoundError(f"No word images found in {image_dir}")

    output_index = Path(output_index)
    output_labels = Path(output_labels)
    output_names = Path(output_names)
    output_codebook = Path(output_codebook)
    output_pca = Path(output_pca)
    output_index.parent.mkdir(parents=True, exist_ok=True)

    print("Building classical OCR vector database")
    print(f"  feature: HOG + LBP + dense SIFT BoVW")
    print(f"  PCA dim: {pca_dim}")
    print("  FAISS : IndexFlatIP (exact cosine search after L2 normalization)")
    print()

    sift_paths = [image_dir / f for f in files[: min(sift_train_images, len(files))]]
    print(f"Training dense-SIFT codebook: {sift_words} visual words from {len(sift_paths)} images")
    codebook = fit_sift_codebook(
        sift_paths,
        codebook_size=sift_words,
        max_descriptors=sift_max_descriptors,
    )
    save_pickle(codebook, output_codebook)

    pca_count = min(pca_samples, len(files))
    print(f"Extracting {pca_count} raw vectors for PCA")
    pca_paths = [image_dir / f for f in files[:pca_count]]
    pca_vectors = []
    for start in tqdm(range(0, pca_count, batch_size), desc="PCA sample"):
        batch_paths = pca_paths[start:start + batch_size]
        pca_vectors.append(_extract_batch(batch_paths, codebook))
    pca_vectors = np.vstack(pca_vectors).astype(np.float32)
    print(f"  raw dim: {pca_vectors.shape[1]}")

    print("Training PCA")
    pca_model = fit_pca(pca_vectors, output_dim=pca_dim)
    save_pickle(pca_model, output_pca)

    dim = pca_model.components.shape[0]
    index = faiss.IndexFlatIP(dim)
    indexed_labels: list[str] = []
    indexed_names: list[str] = []

    print(f"Indexing {len(files)} word images")
    for start in tqdm(range(0, len(files), batch_size), desc="Indexing"):
        batch_files = files[start:start + batch_size]
        batch_paths = [image_dir / f for f in batch_files]
        raw = _extract_batch(batch_paths, codebook)
        reduced = pca_model.transform(raw)
        reduced = l2_normalize(reduced, axis=1).astype(np.float32)
        index.add(reduced)

        for fname in batch_files:
            img_id = Path(fname).stem
            indexed_labels.append(labels_dict.get(img_id, "unknown"))
            indexed_names.append(fname)

        del raw, reduced
        gc.collect()

    if index.ntotal != len(indexed_labels):
        raise ValueError(
            f"Index size ({index.ntotal}) does not match labels count ({len(indexed_labels)})."
        )

    faiss.write_index(index, str(output_index))
    np.save(output_labels, np.asarray(indexed_labels, dtype=object))
    np.save(output_names, np.asarray(indexed_names, dtype=object))

    print()
    print(f"Saved index     : {output_index}")
    print(f"Saved labels    : {output_labels}")
    print(f"Saved names     : {output_names}")
    print(f"Saved codebook  : {output_codebook}")
    print(f"Saved PCA model : {output_pca}")
    print(f"Indexed vectors : {index.ntotal}")
    print(f"Vector dim      : {dim}")
    return index, indexed_labels, indexed_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build HOG+LBP+dense-SIFT PCA FAISS index")
    parser.add_argument("--image-dir", default="./Vietnam/train_word")
    parser.add_argument("--csv-path", default="./Vietnam/train_word.csv")
    parser.add_argument("--output-index", default="./models/vector_database.index")
    parser.add_argument("--output-labels", default="./models/word_labels.npy")
    parser.add_argument("--output-names", default="./models/file_names.npy")
    parser.add_argument("--output-codebook", default="./models/codebook.pkl")
    parser.add_argument("--output-pca", default="./models/pca_model.pkl")
    parser.add_argument("--pca-dim", type=int, default=512)
    parser.add_argument("--pca-samples", type=int, default=5000)
    parser.add_argument("--sift-words", type=int, default=128)
    parser.add_argument("--sift-train-images", type=int, default=1500)
    parser.add_argument("--sift-max-descriptors", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional development limit. Omit for the full train_word set.")
    args = parser.parse_args()

    build_vector_database(
        image_dir=args.image_dir,
        csv_path=args.csv_path,
        output_index=args.output_index,
        output_labels=args.output_labels,
        output_names=args.output_names,
        output_codebook=args.output_codebook,
        output_pca=args.output_pca,
        pca_dim=args.pca_dim,
        pca_samples=args.pca_samples,
        sift_words=args.sift_words,
        sift_train_images=args.sift_train_images,
        sift_max_descriptors=args.sift_max_descriptors,
        batch_size=args.batch_size,
        max_images=args.max_images,
    )
