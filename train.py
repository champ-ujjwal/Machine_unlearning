# import torch
# import torch.nn.functional as F
# from torch.optim import Adam
# from tqdm import tqdm
# import os
# import csv

# from utils import load_clip, encode_with_grad, encode_no_grad
# from dataset import get_datasets
# from lora import apply_lora
# # from utils import encode_no_grad, encode_with_grad, load_clip, encode
# from loss import unlearning_loss

# device = "cuda" if torch.cuda.is_available() else "cpu"

# CHECKPOINT_DIR = "checkpoints"
# RESULTS_DIR = "results"
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# os.makedirs(RESULTS_DIR, exist_ok=True)


# # -------------------------------------------------
# # Helper: Evaluation (Accuracy + Recall@k)
# # -------------------------------------------------
# def evaluate(model, dataset, label="Eval"):
#     correct = 0
#     total = 0
#     r1 = 0
#     r5 = 0

#     for img, text in dataset[:500]:  # speed limit
#         z_img, z_txt = encode_no_grad(model, [img], [text], device)

#         sim = (z_img @ z_txt.T).item()

#         if sim > 0.25:
#             correct += 1
#             r1 += 1
#         if sim > 0.15:
#             r5 += 1

#         total += 1

#     acc = correct / total
#     r1 = r1 / total
#     r5 = r5 / total

#     print(f"[{label}] Acc: {acc:.4f} | R@1: {r1:.4f} | R@5: {r5:.4f}")
#     return acc, r1, r5



# # -------------------------------------------------
# # CSV logger
# # -------------------------------------------------
# def log_to_csv(filename, row):
#     file_exists = os.path.isfile(filename)

#     with open(filename, mode='a', newline='') as f:
#         writer = csv.writer(f)

#         if not file_exists:
#             writer.writerow([
#                 "Forget_Class",
#                 "Forget_Acc_Before", "Forget_Acc_After",
#                 "Retain_Acc_Before", "Retain_Acc_After",
#                 "Recall1_Forget_Before", "Recall1_Forget_After",
#                 "Recall1_Retain_Before", "Recall1_Retain_After",
#                 "Recall5_Forget_Before", "Recall5_Forget_After",
#                 "Recall5_Retain_Before", "Recall5_Retain_After"
#             ])

#         writer.writerow(row)


# # -------------------------------------------------
# # Load CLIP
# # -------------------------------------------------
# clip_model, preprocess = load_clip(device)

# # # Freeze text encoder
# # for p in clip_model.transformer.parameters():
# #     p.requires_grad = False

# import copy
# clip_teacher = copy.deepcopy(clip_model).to(device)
# clip_teacher.eval()
# # Freeze entire CLIP
# for p in clip_model.parameters():
#     p.requires_grad = False


# # Apply LoRA to image encoder
# clip_model = apply_lora(clip_model).to(device)

# print("\nAfter LoRA:")
# for n, p in clip_model.named_parameters():
#     if p.requires_grad:
#         print(n)

# print("\nTrainable Parameters (LoRA only):")
# for n, p in clip_model.named_parameters():
#     if p.requires_grad:
#         print(n)

# optimizer = Adam(clip_model.visual.parameters(), lr=1e-5)

# # -------------------------------------------------
# # Dataset split
# # -------------------------------------------------
# FORGET_CLASS = "apple"
# forget, retain = get_datasets(FORGET_CLASS)

# print("\n========== DATASET INFO ==========")
# print("Forget class:", FORGET_CLASS)
# print("Forget samples:", len(forget))
# print("Retain samples:", len(retain))
# print("Example forget label:", forget[0][1])
# print("Example retain label:", retain[0][1])
# print("==================================\n")

# # -------------------------------------------------
# # Test loss grad
# # -------------------------------------------------
# img, text = forget[0]
# z_img_new, z_txt = encode_with_grad(clip_model, [img], [text], device)
# z_img_old, _ = encode_no_grad(clip_teacher, [img], [text], device)
# test_loss = unlearning_loss(z_img_new, z_txt, z_img_old)
# print(f"Test loss requires grad: {test_loss.requires_grad}")
# print(f"Test loss: {test_loss.item()}")

# # -------------------------------------------------
# # Evaluation BEFORE training
# # -------------------------------------------------
# print("\nEvaluating BEFORE unlearning...")
# forget_before = evaluate(clip_model, forget, "Forget BEFORE")
# retain_before = evaluate(clip_model, retain, "Retain BEFORE")

# # -------------------------------------------------
# # Training Loop
# # -------------------------------------------------
# EPOCHS = 3   # For testing purposes, set to 1. Increase as needed. otherwise. 5 

# for epoch in range(EPOCHS):
#     clip_model.train()
#     total_loss = 0

#     for step, ((img, text), (img_r, text_r)) in enumerate(tqdm(zip(forget[:500], retain[:500]))):
#         # Teacher embedding (no LoRA) for forget
#         z_img_old, z_txt = encode_no_grad(clip_teacher, [img], [text], device)

#         # Student embedding (LoRA, with grad) for forget
#         z_img_new, _ = encode_with_grad(clip_model, [img], [text], device)

#         # ---------- Forget Loss ----------
#         sim = (z_img_new * z_txt).sum(dim=-1)
#         L_forget = (sim ** 2).mean()

#         # ---------- Perturb Loss ----------
#         L_perturb = ((z_img_new - z_img_old) ** 2).mean()

#         # ---------- Distill Loss for Retain ----------
#         z_img_old_r, z_txt_r = encode_no_grad(clip_teacher, [img_r], [text_r], device)
#         z_img_new_r, _ = encode_with_grad(clip_model, [img_r], [text_r], device)
#         with torch.no_grad():
#             p_old_r = F.softmax(z_img_old_r @ z_txt_r.T, dim=-1)
#         p_new_r = F.log_softmax(z_img_new_r @ z_txt_r.T, dim=-1)
#         L_distill_r = F.kl_div(p_new_r, p_old_r, reduction="batchmean")

#         loss = 1 * L_forget + 5 * L_perturb + 5 * L_distill_r

#         print(f"Loss requires grad: {loss.requires_grad}")
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#         if step % 50 == 0:
#             print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

#     print(f"Epoch {epoch} finished | Total Loss: {total_loss:.4f}")

#     # Save checkpoint
#     torch.save(
#         clip_model.state_dict(),
#         f"{CHECKPOINT_DIR}/epoch_{epoch}.pt"
#     )

# # -------------------------------------------------
# # Evaluation AFTER training
# # -------------------------------------------------
# print("\nEvaluating AFTER unlearning...")
# forget_after = evaluate(clip_model, forget, "Forget AFTER")
# retain_after = evaluate(clip_model, retain, "Retain AFTER")

# print("\n=========== SUMMARY ===========")
# print("Forget Accuracy BEFORE:", forget_before[0])
# print("Forget Accuracy AFTER :", forget_after[0])
# print("Retain Accuracy BEFORE:", retain_before[0])
# print("Retain Accuracy AFTER :", retain_after[0])
# print("================================")

# # -------------------------------------------------
# # Save results to CSV
# # -------------------------------------------------
# csv_file = os.path.join(RESULTS_DIR, "metrics.csv")

# log_to_csv(csv_file, [
#     FORGET_CLASS,
#     forget_before[0], forget_after[0],
#     retain_before[0], retain_after[0],
#     forget_before[1], forget_after[1],
#     retain_before[1], retain_after[1],
#     forget_before[2], forget_after[2],
#     retain_before[2], retain_after[2],
# ])

# print(f"\nResults saved to {csv_file}")































# import torch
# from torch.optim import Adam
# from tqdm import tqdm
# import copy

# from utils import load_clip, encode_with_grad, encode_no_grad
# from dataset import get_datasets
# from lora import apply_lora
# from loss import forget_loss, perturb_loss, distill_loss

# device = "cuda" if torch.cuda.is_available() else "cpu"

# # ---------------- Load CLIP ----------------
# clip_model = load_clip(device)
# clip_teacher = copy.deepcopy(clip_model).to(device)
# clip_teacher.eval()

# # Freeze whole model
# for p in clip_model.parameters():
#     p.requires_grad = False

# # Apply LoRA to visual encoder
# clip_model = apply_lora(clip_model).to(device)

# optimizer = Adam(clip_model.visual.parameters(), lr=1e-5)

# # ---------------- Dataset ----------------
# FORGET_CLASS = "apple"
# forget, retain = get_datasets(FORGET_CLASS)

# print("Forget samples:", len(forget))
# print("Retain samples:", len(retain))

# # ---------------- Training ----------------
# EPOCHS = 1

# for epoch in range(EPOCHS):
#     clip_model.train()
#     total_loss = 0

#     for (img_f, text_f), (img_r, text_r) in tqdm(zip(forget[:500], retain[:500])):

#         # ===== Forget branch =====
#         z_img_old_f, z_txt_f = encode_no_grad(clip_teacher, [img_f], [text_f], device)
#         z_img_new_f, _ = encode_with_grad(clip_model, [img_f], [text_f], device)

#         L_forget = forget_loss(z_img_new_f, z_txt_f)

#         # ===== Retain branch =====
#         z_img_old_r, z_txt_r = encode_no_grad(clip_teacher, [img_r], [text_r], device)
#         z_img_new_r, _ = encode_with_grad(clip_model, [img_r], [text_r], device)

#         L_perturb = perturb_loss(z_img_new_r, z_img_old_r)
#         L_distill = distill_loss(z_img_new_r, z_img_old_r, z_txt_r)

#         loss = L_forget + 5 * L_perturb + 5 * L_distill

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch} | Loss: {total_loss:.4f}")




































import torch
from torch.optim import Adam
from tqdm import tqdm
import copy

from utils import load_clip, encode_with_grad, encode_no_grad
from dataset import get_datasets
from lora import apply_lora
from loss import contrastive_retain_loss, forget_loss, perturb_loss, distill_loss

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------
# Evaluation Function
# -------------------------------------------------
def evaluate(model, dataset, label="Eval"):
    correct = 0
    total = 0

    for img, text in dataset[:500]:
        z_img, z_txt = encode_no_grad(model, [img], [text], device)
        sim = (z_img @ z_txt.T).item()

        if sim > 0.25:
            correct += 1
        total += 1

    acc = correct / total
    print(f"[{label}] Accuracy: {acc:.4f}")
    return acc


# -------------------------------------------------
# Load CLIP
# -------------------------------------------------
clip_model = load_clip(device)

# Teacher (original frozen CLIP)
clip_teacher = copy.deepcopy(clip_model).to(device)
clip_teacher.eval()

# Freeze whole CLIP
for p in clip_model.parameters():
    p.requires_grad = False

# Apply LoRA to visual encoder
clip_model = apply_lora(clip_model).to(device)

print("\nTrainable parameters after LoRA:")
for n, p in clip_model.named_parameters():
    if p.requires_grad:
        print(n)

optimizer = Adam(clip_model.visual.parameters(), lr=1e-5)


# -------------------------------------------------
# Dataset
# -------------------------------------------------
FORGET_CLASS = "apple"
forget, retain = get_datasets(FORGET_CLASS)

print("\nForget samples:", len(forget))
print("Retain samples:", len(retain))


# -------------------------------------------------
# Evaluate BEFORE training
# -------------------------------------------------
print("\n==== BEFORE UNLEARNING ====")
forget_before = evaluate(clip_model, forget, "Forget BEFORE")
retain_before = evaluate(clip_model, retain, "Retain BEFORE")


# -------------------------------------------------
# Training Loop
# -------------------------------------------------
EPOCHS = 1

for epoch in range(EPOCHS):
    clip_model.train()
    total_loss = 0

    for (img_f, text_f), (img_r, text_r) in tqdm(
        zip(forget[:500], retain[:500]), total=500
    ):

        # -------- FORGET BRANCH --------
        z_img_old_f, z_txt_f = encode_no_grad(
            clip_teacher, [img_f], [text_f], device
        )
        z_img_new_f, _ = encode_with_grad(
            clip_model, [img_f], [text_f], device
        )

        L_forget = forget_loss(z_img_new_f, z_txt_f)

        # -------- RETAIN BRANCH --------
        z_img_old_r, z_txt_r = encode_no_grad(
            clip_teacher, [img_r], [text_r], device
        )
        z_img_new_r, _ = encode_with_grad(
            clip_model, [img_r], [text_r], device
        )


        L_contrast = contrastive_retain_loss(z_img_new_r, z_txt_r)
        L_perturb = perturb_loss(z_img_new_r, z_img_old_r)
        L_distill = distill_loss(z_img_new_r, z_img_old_r, z_txt_r)

        # -------- Total Loss --------
        # loss = L_forget + 5 * L_perturb + 5 * L_distill
        loss = (
                     0.05 * L_forget
                     + 20 * L_perturb
                     + 20 * L_distill
                     + 10 * L_contrast
                )



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"\nEpoch {epoch} | Total Loss: {total_loss:.4f}")


# -------------------------------------------------
# Evaluate AFTER training
# -------------------------------------------------
print("\n==== AFTER UNLEARNING ====")
forget_after = evaluate(clip_model, forget, "Forget AFTER")
retain_after = evaluate(clip_model, retain, "Retain AFTER")

print("\n=========== SUMMARY ===========")
print("Forget BEFORE :", forget_before)
print("Forget AFTER  :", forget_after)
print("Retain BEFORE :", retain_before)
print("Retain AFTER  :", retain_after)
print("================================")
