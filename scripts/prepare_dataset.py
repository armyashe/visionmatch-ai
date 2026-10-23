import csv
from collections import Counter # dùng để đếm số lượng các phần tử trong một iterable, ví dụ như danh sách hoặc chuỗi.
from pathlib import Path # là thư viện để làm việc với đường dẫn tệp và thư mục trong Python.


# là thư viện để xử lý hình ảnh trong Python. 
# Nó cung cấp các lớp và phương thức để mở, chỉnh sửa và lưu hình ảnh ở nhiều định dạng khác nhau.    
from PIL import Image, UnidentifiedImageError 


# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CATALOG_PATH = PROCESSED_DIR / "catalog.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_images():
    """Find all image files recursively."""
    print("[1/3] Collecting images...")

    images = sorted(
        path
        for path in RAW_DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Found images: {len(images)}")
    return images


def prepare_catalog(images):
    """Validate images and build catalog records."""
    print("\n[2/3] Validating images...")

    records = []
    invalid_images = []

    for image_path in images:
        try:
            with Image.open(image_path) as img:
                img.verify()

            # Check that image pixels can be decoded
            with Image.open(image_path) as img:
                img.load()
                width, height = img.size

            category = image_path.parent.name

            record = {
                "image_id": len(records),
                "image_path": image_path.relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "category": category,
                "width": width,
                "height": height,
            }

            records.append(record)

        except (OSError, ValueError, UnidentifiedImageError) as error:
            invalid_images.append((str(image_path), str(error)))

    print(f"Valid images: {len(records)}")
    print(f"Invalid images: {len(invalid_images)}")

    for path, error in invalid_images[:10]:
        print(f"INVALID: {path} - {error}")

    return records


def save_catalog(records):
    """Save validated metadata to CSV."""
    print("\n[3/3] Saving catalog...")

    if not records:
        raise RuntimeError("No valid images found.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_id",
        "image_path",
        "category",
        "width",
        "height",
    ]

    with CATALOG_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Catalog saved: {CATALOG_PATH}")
    print(f"Total records: {len(records)}")

    counts = Counter(row["category"] for row in records)

    print("\nImages per category:")
    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")

    print("\nSample records:")
    for record in records[:5]:
        print(record)


def main():
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {RAW_DATA_DIR}"
        )

    images = collect_images()
    records = prepare_catalog(images)
    save_catalog(records)

    print("\nDataset preparation completed!")


if __name__ == "__main__":
    main()
