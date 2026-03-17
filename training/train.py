import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import json
from datetime import datetime

from tqdm import tqdm

from loader import PopulationDataset
from base_model import base_model


def train(dataset_dir, output_dir,
    lr: float, weight_decay: float, num_epochs: int,
    dropout_prob: float, train_body: bool
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model on device: '{device}'")

    model = base_model(dropout_prob, train_body)
    model = model.to(device)

    train_dataset = PopulationDataset(
        dir=dataset_dir,
        split="train",
    )
    val_dataset = PopulationDataset(
        dir=dataset_dir,
        split="val",
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": []}

    try:
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
    
    except KeyboardInterrupt:
        print("Training interrupted... Saving model.")

    # Make a directory for storing model stuff
    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_dir = os.path.join(output_dir, timestamp)
    os.makedirs(model_dir, exist_ok=True)

    # Save general info about training
    config_text = f"""
    Training Configuration
    ----------------------
    learning_rate   : {lr:.4e}
    weight_decay    : {weight_decay:.4e}
    num_epochs      : {num_epochs}
    dropout_prob    : {dropout_prob:.4e}
    train_body      : {train_body}
    """.strip()
    with open(os.path.join(model_dir, "training_info.txt"), "w") as f:
        f.write(config_text)
    
    # Save the model itself
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))

    # Save the history of training the model
    with open(os.path.join(model_dir, "training_history.json"), "w") as f:
        json.dump(history, f)
