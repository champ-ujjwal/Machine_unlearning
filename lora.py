from peft import LoraConfig, get_peft_model

def apply_lora(clip_model):
    """
    Apply LoRA to OpenAI CLIP ViT attention layers correctly.
    """

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["out_proj", "c_fc", "c_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    clip_model.visual = get_peft_model(clip_model.visual, config)
    return clip_model
