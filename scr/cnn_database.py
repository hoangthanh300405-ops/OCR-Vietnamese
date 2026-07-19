from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.cnn_features import EMBED_DIM, WordEmbeddingNet, load_word_image
from src.database import _word_files, load_labels
from src.features import save_pickle


class _InferDataset(Dataset):
    def __init__(self, image_dir: Path, files: list[str]):
        self.image_dir = image_dir
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        try:
            return load_word_image(self.image_dir / self.files[idx])
        except Exception:
            return torch.zeros(1, 256, 128)


def build_cnn_database(
    image_dir: str | Path = "./Vietnam/train_word",
    csv_path: str | Path = "./Vietnam/train_word.csv",
    model_path: str | Path = "./models/cnn_embedding.pt",
    output_index: str | Path = "./models/cnn_vector_database.index",
    output_labels: str | Path = "./models/cnn_word_labels.npy",
    output_names: str | Path = "./models/cnn_file_names.npy",
    batch_size: int = 256,
    num_workers: int = 4,
    max_images: int | None = None,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = Path(image_dir)
    files = _word_files(image_dir)
    if max_images is not None:
        files = files[:max_images]
    if not files:
        raise FileNotFoundError(f"No word images found in {image_dir}")

    labels_dict = load_labels(csv_path)

    # ── Load model ──────────────────────────────────────────────────────────
    checkpoint = torch.load(model_path, map_location=device)
    embed_dim = checkpoint.get("embed_dim", EMBED_DIM)
    model = WordEmbeddingNet(embed_dim=embed_dim, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded CNN model  : {model_path}  (embed_dim={embed_dim}, device={device})")

    # ── FAISS index ──────────────────────────────────────────────────────────
    # IndexHNSWFlat gives sub-linear approximate search at query time while
    # keeping build cost low. Tune M (graph degree) vs memory as needed.
    index = faiss.IndexHNSWFlat(embed_dim, 32)
    index.hnsw.efConstruction = 200  # higher = better graph, slower build

    indexed_labels: list[str] = []
    indexed_names: list[str] = []

    dataset = _InferDataset(image_dir, files)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    print(f"Indexing {len(files)} word images …")
    offset = 0
    with torch.no_grad():
        for batch_tensors in tqdm(loader, desc="Indexing"):
            batch_tensors = batch_tensors.to(device)
            embeddings = model(batch_tensors).cpu().numpy().astype(np.float32)
            # IndexHNSWFlat uses L2 internally; since embeddings are L2-normalised
            # by WordEmbeddingNet (F.normalize), L2-distance rank == cosine rank.
            index.add(embeddings)

            batch_files = files[offset: offset + len(embeddings)]
            for fname in batch_files:
                img_id = Path(fname).stem
                indexed_labels.append(labels_dict.get(img_id, "unknown"))
                indexed_names.append(fname)
            offset += len(embeddings)
            del batch_tensors, embeddings
            gc.collect()

    # ── Save artefacts ────────────────────────────────────────────────────────
    for p in [output_index, output_labels, output_names]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_index))
    np.save(output_labels, np.asarray(indexed_labels, dtype=object))
    np.save(output_names, np.asarray(indexed_names, dtype=object))

    print(f"\nSaved index   : {output_index}")
    print(f"Saved labels  : {output_labels}")
    print(f"Saved names   : {output_names}")
    print(f"Total vectors : {index.ntotal}  (dim={embed_dim})")
    return index, indexed_labels, indexed_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CNN embedding FAISS index (ResNet18 + ArcFace)")
    parser.add_argument("--image-dir", default="./Vietnam/train_word")
    parser.add_argument("--csv-path", default="./Vietnam/train_word.csv")
    parser.add_argument("--model-path", default="./models/cnn_embedding.pt")
    parser.add_argument("--output-index", default="./models/cnn_vector_database.index")
    parser.add_argument("--output-labels", default="./models/cnn_word_labels.npy")
    parser.add_argument("--output-names", default="./models/cnn_file_names.npy")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    build_cnn_database(
        image_dir=args.image_dir,
        csv_path=args.csv_path,
        model_path=args.model_path,
        output_index=args.output_index,
        output_labels=args.output_labels,
        output_names=args.output_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_images=args.max_images,
    )
