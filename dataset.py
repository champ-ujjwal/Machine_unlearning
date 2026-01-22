import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms

def get_datasets(forget_class="apple"):
    """
    Loads CIFAR100 and splits into forget and retain sets.
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    dataset = CIFAR100(root="./data", download=True, transform=transform)

    # Correct way to get class names
    LABELS = dataset.classes
    forget_idx = LABELS.index(forget_class)

    forget, retain = [], []

    for img, label in dataset:
        text = LABELS[label]
        if label == forget_idx:
            forget.append((img, text))
        else:
            retain.append((img, text))

    return forget, retain
