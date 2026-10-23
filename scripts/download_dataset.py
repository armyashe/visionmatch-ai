from pathlib import Path
from zipfile import ZipFile, BadZipFile

from huggingface_hub import hf_hub_download


# Configuration
REPO_ID = "Cleanlab/footwear-demo"
FILENAME = "footwear-demo.zip"

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset():
    """Download the dataset from Hugging Face."""
    print("[1/3] Downloading dataset...")

    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="dataset",
    )

    print(f"Dataset ZIP: {zip_path}")
    return Path(zip_path)


def extract_dataset(zip_path):
    """Extract the dataset into data/raw/."""
    print("\n[2/3] Extracting dataset...")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path, "r") as zip_file:
        members = zip_file.infolist()
        print(f"ZIP entries: {len(members)}")

        # Validate paths before extraction
        root = RAW_DATA_DIR.resolve()
        for member in members:
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Unsafe ZIP entry: {member.filename}")

        zip_file.extractall(root)

    print(f"Extracted to: {root}")


def verify_dataset():
    """Count extracted images and display samples."""
    print("\n[3/3] Verifying dataset...")

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    images = sorted(
        path
        for path in RAW_DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in image_extensions
    )

    print(f"Total images: {len(images)}")

    for image_path in images[:10]:
        print(" -", image_path.relative_to(PROJECT_ROOT))

    if not images:
        raise RuntimeError("No images found in data/raw/")

    print("\nDataset download and extraction completed!")


def main():
    try:
        zip_path = download_dataset()
        extract_dataset(zip_path)
        verify_dataset()
    except (OSError, BadZipFile, ValueError, RuntimeError) as error:
        print(f"\nDataset setup failed: {error}")
        raise


if __name__ == "__main__":
    main()
