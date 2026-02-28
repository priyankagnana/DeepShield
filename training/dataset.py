from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(num_workers: int = 4):
    # Frames are already saved at 224×224 by frame_extractor — no Resize needed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder("data/processed/train", transform)
    val_dataset = datasets.ImageFolder("data/processed/val", transform)
    test_dataset = datasets.ImageFolder("data/processed/test", transform)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader