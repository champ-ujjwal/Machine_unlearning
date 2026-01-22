import torch
from PIL import Image
from utils import load_clip, encode_no_grad, PREPROCESS

# device = "cuda"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


CHECKPOINT_DIR = "checkpoints"   # change if different
image_path = "images/bat.jpeg"    # change to your test image

texts = [["bat"], ["cricket"], ["batman"]]

# 1️⃣ Load model architecture
clip_model, preprocess = load_clip(device)
clip_model.eval()

# 2️⃣ Prepare image once
img = Image.open(image_path).convert("RGB")
img_tensor = preprocess(img)

for epoch in range(4):
    print(f"\n===== Evaluating epoch_{epoch}.pt =====")

    ckpt_path = f"{CHECKPOINT_DIR}/epoch_{epoch}.pt"
    state_dict = torch.load(ckpt_path, map_location=device)
    clip_model.load_state_dict(state_dict)
    clip_model.eval()

    # image must be list because utils expects list
    images = [img_tensor]

    for t in texts:
        z_img, z_txt = encode_no_grad(clip_model, images, t, device)
        sim = torch.cosine_similarity(z_img, z_txt).item()
        print(f"{t[0]}: {sim:.4f}")
