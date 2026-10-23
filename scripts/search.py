"""
Search for similar footwear images using CLIP embeddings.
This script loads a precomputed index of image embeddings and allows you to query
the index with a new image to find visually similar images.
Usage:
    python search.py --image path/to/query_image.jpg --k 5
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_ROOT / "data/processed/catalog.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "artifacts/clip_embeddings.npy"
MODEL_ID = "openai/clip-vit-base-patch32"


def load_index():
    """Load catalog and precomputed CLIP embeddings."""
    print("[1/4] Loading image index...")

    with CATALOG_PATH.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        records = list(csv.DictReader(file))

    embeddings = np.load(
        EMBEDDINGS_PATH, allow_pickle=False
    )

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")

    if embeddings.shape != (len(records), 512):
        raise ValueError(
            "Embedding shape does not match catalog/model."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Index contains NaN or Inf.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Index is not L2 normalized.")

    print(f"Catalog images: {len(records)}")
    print(f"Embeddings shape: {embeddings.shape}")

    return records, embeddings


def load_model():
    """Load the same CLIP used for indexing."""
    print("\n[2/4] Loading CLIP...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID)
    model = model.to(device)
    model.eval()

    print(f"Device: {device}")

    return processor, model, device


def encode_query(image_path, processor, model, device):
    """Convert the query image to a normalized vector."""
    print("\n[3/4] Encoding query image...")

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

    inputs = processor(
        images=rgb_image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        features = model.get_image_features(**inputs)

        if not isinstance(features, torch.Tensor):
            features = features.pooler_output

        features = torch.nn.functional.normalize(
            features, p=2, dim=-1
        )

    query_embedding = features.cpu().numpy()[0]

    return query_embedding


def search(query_embedding, records, embeddings, query_path, k):
    """Rank catalog images by cosine similarity."""
    print("\n[4/4] Searching similar images...")

    scores = embeddings @ query_embedding

    # Exclude exact same file when query is in catalog.
    query_resolved = query_path.resolve()

    candidates = []

    for index, record in enumerate(records):
        catalog_path = (
            PROJECT_ROOT / record["image_path"]
        ).resolve()

        if catalog_path == query_resolved:
            continue

        candidates.append(index)

    ranked_indices = sorted(
        candidates,
        key=lambda index: float(scores[index]),
        reverse=True
    )[:k]

    results = []

    for rank, index in enumerate(ranked_indices, start=1):
        record = records[index]

        results.append({
            "rank": rank,
            "score": float(scores[index]),
            "category": record["category"],
            "image_path": record["image_path"]
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Search similar footwear images using CLIP"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the query image"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results"
    )

    args = parser.parse_args()

    query_path = Path(args.image)

    if not query_path.is_absolute():
        query_path = PROJECT_ROOT / query_path

    if not query_path.is_file():
        parser.error(f"Query image not found: {query_path}")

    if args.k < 1:
        parser.error("--k must be greater than zero")

    records, embeddings = load_index()
    processor, model, device = load_model()

    query_embedding = encode_query(
        query_path, processor, model, device
    )

    results = search(
        query_embedding,
        records,
        embeddings,
        query_path,
        args.k
    )

    print(f"\nQuery: {query_path}")
    print(f"\nTop-{len(results)} similar images:\n")

    for result in results:
        print(
            f"{result['rank']}. "
            f"Score: {result['score']:.4f} | "
            f"Category: {result['category']} | "
            f"Path: {result['image_path']}"
        )


if __name__ == "__main__":
    main()
