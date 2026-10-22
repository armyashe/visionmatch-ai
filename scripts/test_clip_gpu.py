
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

image = Image.new("RGB", (224, 224), color="white")
inputs = processor(images=image, return_tensors="pt").to(DEVICE)

with torch.inference_mode():
    embedding = model.get_image_features(**inputs)

    # Compatible with Transformers versions returning ModelOutput
    if hasattr(embedding, "pooler_output"):
        embedding = embedding.pooler_output

    embedding = torch.nn.functional.normalize(embedding, dim=-1)

print("Embedding shape:", embedding.shape)
print("Embedding device:", embedding.device)
print("L2 norm:", embedding.norm(dim=-1).item())
