import clip
from transformers import AutoImageProcessor, AutoModel


def load_clip():
    print("Loading CLIP ViT-B/32...")
    clip.load("ViT-B/32", device="cpu")
    print("Loading CLIP ViT-L/14...")
    clip.load("ViT-L/14", device="cpu")


def load_dinov3():
    model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"
    print(f"Loading DINOv3 {model_id}...")
    AutoImageProcessor.from_pretrained(model_id)
    AutoModel.from_pretrained(model_id)


if __name__ == "__main__":
    load_clip()
    load_dinov3()
    print("Done.")
