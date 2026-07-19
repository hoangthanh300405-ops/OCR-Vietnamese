# Vietnamese Handwritten OCR — Segmentation & Vector-Search Retrieval

A retrieval-based OCR system for handwritten Vietnamese text. Instead of a
sequence decoder, the system **segments** a line/paragraph image into
individual word crops, **embeds** each crop as a vector, and **retrieves**
the closest match from a pre-indexed vocabulary using FAISS similarity
search. Two interchangeable embedding backends are provided: a classical
hand-engineered pipeline and a CNN (ResNet18 + ArcFace) pipeline.

---

## Outline

1. [Overview](#overview)
2. [Pipeline stages](#pipeline-stages)
   - Segmentation
   - Feature extraction (classical vs. CNN)
   - Vector database & retrieval
3. [Project structure](#project-structure)
4. [Two embedding backends](#two-embedding-backends)
5. [Usage](#usage)
   - Building the index
   - Training the CNN embedding
   - Recognizing an image
   - Validation / evaluation
6. [Tech stack](#tech-stack)
7. [Data](#data)

---

## Overview

Given a photo or scan of handwritten Vietnamese text (e.g. *"tôi tên là
linh"*), the system:

1. Splits the image into per-word crops.
2. Converts each crop into a fixed-length vector.
3. Searches a FAISS index of pre-labeled reference vectors for the closest
   match(es).
4. Returns the label of the best (or majority-voted) match as the
   recognized word.

Because recognition is nearest-neighbor retrieval rather than sequence
decoding, adding new vocabulary only requires re-indexing labeled crops —
no retraining of a classifier is strictly required for the classical
pipeline, and only the embedding backbone needs training for the CNN
pipeline.

## Pipeline stages

### 1. Segmentation
`segmentation/app.py` handles turning a raw photo into individual word
boxes:
- Background normalization (median-blur division) + CLAHE to handle uneven
  lighting on photographed paper.
- Otsu thresholding on the normalized image, with ruled-line/underline
  removal via horizontal morphological opening.
- Connected-component filtering to drop noise.
- Directional dilation sized relative to median stroke height, so letters
  and diacritics merge into word-shaped blobs.
- A lightweight custom DBSCAN clusters nearby fragments (e.g. dotted "i",
  detached diacritics) into single word boxes.
- Boxes are padded and sorted into natural reading order (line, then
  left-to-right).

### 2. Feature extraction — two interchangeable backends
Every segmented crop goes through a shared preprocessing step
(`src/preprocessing.py`: sharpen → CLAHE → adaptive+Otsu binarization →
morphological cleanup → deskew → resize → distance transform) before being
embedded by **either**:

- **Classical (`src/features.py`)** — HOG + uniform LBP + dense-SIFT
  Bag-of-Visual-Words, concatenated and L2-normalized, then reduced with a
  custom dual-PCA to 512-D.
- **CNN (`src/cnn_features.py`)** — a ResNet18 backbone (ImageNet-pretrained,
  first conv adapted to 1-channel input) with an ArcFace margin head used
  only during training, projecting to a 256-D L2-normalized embedding.

### 3. Vector database & retrieval
- **Classical:** `src/database.py` builds an exact-search `IndexFlatIP`
  (cosine similarity via inner product on normalized vectors).
- **CNN:** `src/cnn_database.py` builds an approximate `IndexHNSWFlat` for
  sub-linear query time at scale.
- Both apps (`segmentation/app.py`, `cnn_app.py`) query top-*k* neighbors
  and apply a similarity-weighted majority vote across neighbor labels to
  pick the final predicted word.

## Project structure

```text
├── Vietnam/                     # train/validation/test word crops + label CSVs
├── models/                      # generated artifacts (indexes, codebooks, weights)
├── segmentation/
│   └── app.py                   # segmentation + classical-pipeline recognition CLI
├── src/
│   ├── preprocessing.py         # shared: deskew, binarize, distance transform
│   ├── features.py              # classical HOG+LBP+SIFT+PCA feature extraction
│   ├── database.py              # build classical FAISS IndexFlatIP database
│   ├── recognition.py           # single-crop classical query helper
│   ├── validation.py            # classical pipeline top-k accuracy/precision/recall
│   ├── cnn_features.py          # WordEmbeddingNet (ResNet18) + ArcMarginProduct
│   ├── cnn_database.py          # build CNN FAISS IndexHNSWFlat database
│   ├── cnn_app.py                # CNN-pipeline segmentation + recognition CLI
│   └── cnn_validation.py        # CNN pipeline top-k accuracy/precision/recall
├── train_cnn.py                 # train the ResNet18 + ArcFace embedding model
├── requirements.txt
├── data.md                      # dataset download link
└── README.md
```

## Two embedding backends

| | Classical (HOG+LBP+SIFT+PCA) | CNN (ResNet18+ArcFace) |
|---|---|---|
| Feature extraction | `src/features.py` | `src/cnn_features.py` |
| Requires training | Only PCA + SIFT codebook fitting | Full backbone training (`train_cnn.py`) |
| Index type | `faiss.IndexFlatIP` (exact) | `faiss.IndexHNSWFlat` (approximate) |
| Build script | `src/database.py` | `src/cnn_database.py` |
| Recognition CLI | `segmentation/app.py` | `cnn_app.py` |
| Validation | `src/validation.py` | `cnn_validation.py` |

Both backends share the same segmentation code (`segmentation/app.py`'s
`preprocess_for_segmentation`, `segment_words`, `crop_word`) and the same
image preprocessing (`src/preprocessing.py`), so they can be compared
head-to-head on identical word crops.

## Usage

### Build the classical vector database
```bash
python -m src.database \
  --image-dir ./Vietnam/train_word \
  --csv-path ./Vietnam/train_word.csv \
  --output-index ./models/vector_database.index
```

### Train the CNN embedding, then build its database
```bash
python train_cnn.py --image-dir ./Vietnam/train_word --csv-path ./Vietnam/train_word.csv
python -m src.cnn_database --image-dir ./Vietnam/train_word --csv-path ./Vietnam/train_word.csv
```

### Recognize a full sentence image
```bash
# classical pipeline
python -m segmentation.app --image ./sample.jpg

# CNN pipeline
python cnn_app.py --image ./sample.jpg
```

### Evaluate on the validation set
```bash
python -m src.validation
python cnn_validation.py
```

## Tech stack
- **Image processing / segmentation:** OpenCV (morphology, Otsu, CLAHE), a
  custom lightweight DBSCAN.
- **Classical features:** HOG, uniform LBP, dense SIFT + k-means codebook,
  dual PCA.
- **Deep features:** PyTorch, torchvision ResNet18, ArcFace margin loss.
- **Vector search:** FAISS (`IndexFlatIP` and `IndexHNSWFlat`).
- **Metric:** cosine similarity (via inner product on L2-normalized vectors).

## Data
Sample images and reference/validation word-crop datasets are hosted on
Google Drive — see [`data.md`](./data.md) for the link.
