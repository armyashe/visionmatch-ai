#  file này dùng để xây dựng chỉ mục CLIP từ tập dữ liệu hình ảnh đã chuẩn bị. Nó thực hiện các bước sau:
# 1. Tải catalog.csv chứa siêu dữ liệu hình ảnh.
# 2. Tải mô hình CLIP đã được huấn luyện trước và bộ xử lý tương ứng.
# 3. Trích xuất các embedding hình ảnh từ mô hình CLIP.
# Mục tiêu là tạo ra một tệp NumPy chứa các embedding L2 chuẩn hóa, 
# -->có thể được sử dụng cho các tác vụ tìm kiếm hình ảnh hoặc truy vấn tương tự.

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel


# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_ROOT / "data/processed/catalog.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_ID = "openai/clip-vit-base-patch32"
BATCH_SIZE = 16

#  hàm Đọc danh sách 909 ảnh theo đúng thứ tự trong CSV
def load_catalog():
    """Load image metadata from catalog.csv."""
    print("[1/4] Loading catalog...")

    with CATALOG_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as f:
        records = list(csv.DictReader(f))

    if not records:
        raise ValueError("Catalog is empty.")

    for record in records:
        image_path = PROJECT_ROOT / record["image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

    print(f"Total images: {len(records)}")
    return records

# hàm Nạp pretrained CLIP và đưa lên GPU nếu CUDA khả dụng
# là một mô hình học sâu được huấn luyện trước để trích xuất các đặc trưng từ hình ảnh và văn bản. 
# Nó có thể được sử dụng cho các tác vụ như tìm kiếm hình ảnh, phân loại hình ảnh, và nhiều ứng dụng khác liên quan 
def load_model():
    """Load pretrained CLIP on GPU if available."""
    print("\n[2/4] Loading CLIP model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID)

    model = model.to(device)
    model.eval()

    print(f"Device: {device}")
    print(f"Model: {MODEL_ID}")

    return processor, model, device

#  Xử lý ảnh theo batch 16 ảnh, lấy vector và chuẩn hóa L2
def build_embeddings(records, processor, model, device):
    """Extract L2-normalized image embeddings."""
    print("\n[3/4] Building embeddings...")

    all_embeddings = []

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        images = []

        for record in batch:
            image_path = PROJECT_ROOT / record["image_path"]

            with Image.open(image_path) as img:
                images.append(img.convert("RGB"))

        inputs = processor(
            images=images,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # Tắt gradient để tiết kiệm bộ nhớ khi inference 
        with torch.inference_mode():
            features = model.get_image_features(**inputs)

            # Compatible with Transformers versions that return
            # a model output object instead of a tensor.
            if not isinstance(features, torch.Tensor):
                features = features.pooler_output

            features = torch.nn.functional.normalize(
                features, p=2, dim=-1
            )

        embeddings = features.cpu().numpy()
        all_embeddings.append(embeddings)

        completed = min(start + BATCH_SIZE, len(records))
        print(f"Processed: {completed}/{len(records)}")

    return np.concatenate(all_embeddings, axis=0)

#  Lưu ma trận vector thành file .npy
def save_embeddings(embeddings):
    """Save embeddings as a NumPy array."""
    print("\n[4/4] Saving embeddings...")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = ARTIFACTS_DIR / "clip_embeddings.npy"
    np.save(output_path, embeddings)

    print(f"Saved to: {output_path}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Min L2 norm: {norms.min():.6f}")
    print(f"Max L2 norm: {norms.max():.6f}")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity.")

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embeddings are not L2-normalized.")


def main():
    records = load_catalog()
    processor, model, device = load_model()

    embeddings = build_embeddings(
        records, processor, model, device
    )

    save_embeddings(embeddings)

    print("\nCLIP index building completed!")


if __name__ == "__main__":
    main()
