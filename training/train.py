import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model.cnn_model import DeepfakeCNN
from training.dataset import get_dataloaders
from training.early_stopping import EarlyStopping


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compute_pos_weight(train_loader, device: torch.device) -> torch.Tensor:
    """Compute pos_weight = num_fake / num_real to handle class imbalance."""
    num_fake = 0
    num_real = 0
    for _, labels in train_loader.dataset:
        if labels == 1:
            num_real += 1
        else:
            num_fake += 1
    if num_real == 0:
        return torch.tensor([1.0], device=device)
    ratio = num_fake / num_real
    print(f"Class counts — fake: {num_fake}, real: {num_real}  →  pos_weight: {ratio:.3f}")
    return torch.tensor([ratio], dtype=torch.float, device=device)


def train(max_epochs=50, patience=12, warmup_epochs=5):
    """
    Two-phase fine-tuning of EfficientNet-B0:

    Phase 1 (epochs 1 – warmup_epochs):
      Backbone frozen. Only the classifier head trains at lr=1e-3.
      Fast convergence — teaches the head to map EfficientNet features to
      fake/real without corrupting the pretrained backbone weights.

    Phase 2 (epoch warmup_epochs+1 onwards):
      Last two EfficientNet blocks unfrozen. Full model trains at lr=1e-4
      so the backbone adapts gently to deepfake-specific cues.
    """
    device = _get_device()
    print(f"Using device: {device}")
    os.makedirs("saved_models", exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders()

    model = DeepfakeCNN().to(device)

    pos_weight = _compute_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Phase 1 optimizer — only classifier params are trainable at this point
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
    )

    early_stopping = EarlyStopping(patience=patience, monitor="max")

    best_val_acc = 0.0
    phase = 1

    epoch_bar = tqdm(range(max_epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:

        # Switch to Phase 2 after warm-up
        if phase == 1 and epoch >= warmup_epochs:
            phase = 2
            print(f"\n--- Phase 2: unfreezing top blocks (epoch {epoch+1}) ---")
            model.unfreeze_top_blocks()
            # Rebuild optimizer so newly unfrozen params are included at a
            # lower LR to avoid overwriting pretrained weights aggressively
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-5, weight_decay=1e-4,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=4, min_lr=1e-6
            )

        # Train
        model.train()
        train_loss = 0.0
        train_bar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1}/{max_epochs} [Train|Ph{phase}]",
            leave=False,
        )
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"  Epoch {epoch+1}/{max_epochs} [Val]  ",
                leave=False,
            )
            for images, labels in val_bar:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (outputs.sigmoid() >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        val_loss /= len(val_loader)
        val_accuracy = correct / total if total > 0 else 0.0

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "saved_models/best_model.pth")

        scheduler.step(val_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_accuracy:.4f}",
            lr=f"{current_lr:.2e}",
            phase=phase,
        )
        print(
            f"Epoch {epoch+1}/{max_epochs} [Ph{phase}]  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_accuracy:.4f}  lr={current_lr:.2e}"
        )

        early_stopping(val_accuracy)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}  best_val_acc={best_val_acc:.4f}")
            break

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    train()