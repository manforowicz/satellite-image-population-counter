import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import json

from tqdm import tqdm

from loader import PopulationDataset


def train(dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: '{device}'")

    # pretrained ResNet-50,
    # with final fully connected layer
    # modified to output a single number (log1p of population density)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model = model.to(device)

    dataset_dir = os.path.join(dir, "dataset")

    train_dataset = PopulationDataset(
        dir=dataset_dir,
        split="train",
    )
    val_dataset = PopulationDataset(
        dir=dataset_dir,
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

        print(f"Epoch {epoch}/{num_epochs}")
        for images, targets in tqdm(train_loader):
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

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

    output_dir = os.path.join(dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "my_model.pth"))
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f)
