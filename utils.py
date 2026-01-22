# import clip
# import torch

# # Global holder so all functions can use it
# PREPROCESS = None


# def load_clip(device):
#     global PREPROCESS
#     model, preprocess = clip.load("ViT-B/32", device=device)
#     PREPROCESS = preprocess
#     return model, preprocess


# # Teacher/original model (NO GRAD)
# def encode_no_grad(model, images, texts, device):
#     images = torch.stack(images).to(device)
#     texts = clip.tokenize(texts[0]).to(device)
#     z_img = model.encode_image(images)
#     z_txt = model.encode_text(texts)

#     z_img = z_img / z_img.norm(dim=-1, keepdim=True)
#     z_txt = z_txt / z_txt.norm(dim=-1, keepdim=True)

#     return z_img, z_txt


# # Student/LoRA model (WITH GRAD)
# def encode_with_grad(model, images, texts, device):
#     images = torch.stack(images).to(device)
#     texts = clip.tokenize(texts[0]).to(device)

#     z_img = model.encode_image(images)  # requires grad

#     with torch.no_grad():
#         z_txt = model.encode_text(texts)

#     z_img = z_img / z_img.norm(dim=-1, keepdim=True)
#     z_txt = z_txt / z_txt.norm(dim=-1, keepdim=True)

#     return z_img, z_txt









# import clip
# import torch

# PREPROCESS = None


# def load_clip(device):
#     global PREPROCESS
#     model, preprocess = clip.load("ViT-B/32", device=device)
#     PREPROCESS = preprocess
#     return model


# def encode_no_grad(model, images, texts, device):
#     images = torch.stack(images).to(device)
#     texts = clip.tokenize(texts[0]).to(device)

#     with torch.no_grad():
#         z_img = model.encode_image(images)
#         z_txt = model.encode_text(texts)

#     z_img = z_img / z_img.norm(dim=-1, keepdim=True)
#     z_txt = z_txt / z_txt.norm(dim=-1, keepdim=True)
#     return z_img, z_txt


# def encode_with_grad(model, images, texts, device):
#     images = torch.stack(images).to(device)
#     texts = clip.tokenize(texts[0]).to(device)

#     z_img = model.encode_image(images)  # grad flows
#     with torch.no_grad():
#         z_txt = model.encode_text(texts)

#     z_img = z_img / z_img.norm(dim=-1, keepdim=True)
#     z_txt = z_txt / z_txt.norm(dim=-1, keepdim=True)
#     return z_img, z_txt













import clip
import torch

PREPROCESS = None


def load_clip(device):
    global PREPROCESS
    model, preprocess = clip.load("ViT-B/32", device=device)
    PREPROCESS = preprocess
    return model


def preprocess_if_needed(img):
    # If PIL → apply CLIP preprocess
    if not torch.is_tensor(img):
        return PREPROCESS(img)
    # If already tensor → assume dataset handled it
    return img


def encode_no_grad(model, images, texts, device):
    images = torch.stack([preprocess_if_needed(img) for img in images]).to(device)
    texts = clip.tokenize(texts).to(device)

    with torch.no_grad():
        z_img = model.encode_image(images)
        z_txt = model.encode_text(texts)

    z_img = z_img / z_img.norm(dim=-1, keepdim=True)
    z_txt = z_txt / z_txt.norm(dim=-1, keepdim=True)
    return z_img, z_txt


def encode_with_grad(model, images, texts, device):
    images = torch.stack([preprocess_if_needed(img) for img in images]).to(device)
    texts = clip.tokenize(texts).to(device)

    z_img = model.encode_image(images)
    with torch.no_grad():
        z_txt = model.encode_text(texts)

    z_img = z_img / z_img.norm(dim=-1, keepdim=True)
    z_txt = z_txt / z_txt.norm(dim=-1, keepdim=True)
    return z_img, z_txt


