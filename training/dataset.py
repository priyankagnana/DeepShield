import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(num_workers: int = 4):
    # Frames are already saved at 224×224 by frame_extractor — no Resize needed

    # Augmentation only on the training set to improve generalisation.
    # Deepfake artifacts survive horizontal flips and mild colour/geometric
    # jitter, so these augmentations are safe and won't destroy the signal.
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Val/test: deterministic — no augmentation, just normalise
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder("data/processed/train", train_transform)
    val_dataset   = datasets.ImageFolder("data/processed/val",   eval_transform)
    test_dataset  = datasets.ImageFolder("data/processed/test",  eval_transform)

    # pin_memory is only beneficial on CUDA; skip it on MPS/CPU to avoid the warning
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader