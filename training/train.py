import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import json

from loader import PopulationDataset


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training model on: '{device}'")

    # pretrained ResNet-50,
    # with final fully connected layer
    # modified to output a single number (log1p of population density)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model = model.to(device)

    train_dataset = PopulationDataset(
        dir="/content/cse_493g1_final_project/dataset",
        split="train",
    )
    val_dataset = PopulationDataset(
        dir="/content/cse_493g1_final_project/dataset",
        split="val",
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)  # (batch, 1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device).unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

    os.makedirs("results", exist_ok=True)
    torch.save(model.state_dict(), "results/resnet50_regression.pth")
    with open("results/training_history.json", "w") as f:
        json.dump(history, f)
