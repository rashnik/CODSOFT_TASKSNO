"""
CODSOFT TASK 3 — IMAGE CAPTIONING
Run:
    pip install torch transformers pillow
    python Task3_Image_Captioning.py path/to/image.jpg

The first run downloads the pretrained BLIP model from Hugging Face.
"""
import argparse
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_NAME = "Salesforce/blip-image-captioning-base"

def generate_caption(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=40, num_beams=4)
    return processor.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an AI caption for an image.")
    parser.add_argument("image", help="Path to the image file")
    args = parser.parse_args()
    print("Generated caption:", generate_caption(args.image))
