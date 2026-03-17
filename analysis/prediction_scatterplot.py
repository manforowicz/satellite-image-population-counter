import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from loader import PopulationDataset
from base_model import base_model


def prediction_scatterplot(dataset_dir, model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model on device: '{device}'")

    # Dataset
    dataset = PopulationDataset(dir=dataset_dir, split="test")
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    # Load model
    model_path = os.path.join(model_dir, "model.pth")
    model = base_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preds_log = []
    true_log = []

    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)

            outputs = model(images).squeeze(1).cpu()

            preds_log.extend(outputs.numpy())
            true_log.extend(targets.numpy())
    

    # Convert to numpy
    preds_log = np.array(preds_log)
    true_log = np.array(true_log)

    # Remove empty population data points
    mask = true_log > 0
    preds_log = preds_log[mask]
    true_log = true_log[mask]

    # Convert back from log1p
    pred_density = np.expm1(preds_log)
    true_density = np.expm1(true_log)

    # Plot
    plt.figure(figsize=(6,6))
    plt.scatter(true_density, pred_density, s=8, alpha=0.2)

    # Ideal line
    min_val = true_density.min()
    max_val = true_density.max()
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Ground truth (GHSL)")
    plt.ylabel("Predicted")
    plt.title("True vs predicted density (people/km^2) on test set")
    plt.tight_layout()

    figure_name = os.path.join(model_dir, "prediction_scatterplot.png")
    plt.savefig(figure_name)