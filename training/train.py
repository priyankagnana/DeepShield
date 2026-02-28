import os
import torch
import torch.nn as nn
import torch.optim as optim

from model.cnn_model import DeepfakeCNN
from training.dataset import get_dataloaders
from training.early_stopping import EarlyStopping

def train(max_epochs=50, patience=5):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("saved_models", exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders()

    model = DeepfakeCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(patience=patience)

    best_val_loss = None

    for epoch in range(max_epochs):

        # Train
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (outputs.sigmoid() >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct / total if total > 0 else 0.0

        # Save best model
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "saved_models/best_model.pth")

        print(f"Epoch {epoch+1}/{max_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_accuracy:.4f}")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break


if __name__ == "__main__":
    train()