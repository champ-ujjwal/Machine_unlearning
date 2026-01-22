# import torch
# import torch.nn.functional as F

# def unlearning_loss(z_img_new, z_txt, z_img_old, lambda_f=1, lambda_r=0, lambda_p=1, lambda_d=1):
#     """
#     Implements:

#     L = L_forget + L_retain + L_perturb + L_distill
#     """

#     # ---------- Forget Loss ----------
#     # push similarity to zero
#     sim = (z_img_new * z_txt).sum(dim=-1)
#     L_forget = (sim ** 2).mean()

#     # ---------- Retain Loss (CLIP contrastive style) ----------
#     logits = z_img_new @ z_txt.T
#     labels = torch.arange(len(logits)).to(logits.device)
#     L_retain = F.cross_entropy(logits, labels)

#     # ---------- Perturb Loss ----------
#     L_perturb = ((z_img_new - z_img_old) ** 2).mean()

#     # ---------- Distillation Loss ----------
#     with torch.no_grad():
#         p_old = F.softmax(z_img_old @ z_txt.T, dim=-1)
#     p_new = F.log_softmax(z_img_new @ z_txt.T, dim=-1)
#     L_distill = F.kl_div(p_new, p_old, reduction="batchmean")

#     return (
#         lambda_f * L_forget +
#         lambda_r * L_retain +
#         lambda_p * L_perturb +
#         lambda_d * L_distill
#     )















import torch
import torch.nn.functional as F


def forget_loss(z_img_new, z_txt):
    sim = (z_img_new * z_txt).sum(dim=-1)
    return (sim ** 2).mean()


def perturb_loss(z_img_new, z_img_old):
    return ((z_img_new - z_img_old) ** 2).mean()


def distill_loss(z_img_new, z_img_old, z_txt):
    logits_old = z_img_old @ z_txt.T
    logits_new = z_img_new @ z_txt.T

    p_old = F.softmax(logits_old, dim=-1)
    p_new = F.log_softmax(logits_new, dim=-1)

    return F.kl_div(p_new, p_old, reduction="batchmean")


def contrastive_retain_loss(z_img, z_txt):
    logits = z_img @ z_txt.T
    labels = torch.arange(len(logits)).to(logits.device)
    return F.cross_entropy(logits, labels)
