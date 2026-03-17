import os
import shutil
import torch
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from loader import PopulationDataset
from base_model import base_model

SPLIT = "test"

def save_best_and_worst_images(dataset_dir, model_dir, n_examples: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model on device: '{device}'")

    # Load dataset
    dataset = PopulationDataset(
        dir=dataset_dir, split=SPLIT
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    # Load model
    model_path = os.path.join(model_dir, "model.pth")
    model = base_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = []

    idx = 0

    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)

            preds = model(images).squeeze(1).cpu()

            for i in range(len(preds)):
                row = dataset.data.iloc[idx]

                pred = preds[i].item()
                true = targets[i].item()

                predictions.append({
                    "image_filename": row["image_filename"],
                    "true_log1p_density": true,
                    "pred_log1p_density": pred,
                    "error": pred - true,
                    "abs_error": abs(pred - true)
                })

                idx += 1

    df = pd.DataFrame(predictions)

    # Largest underestimates (prediction << true)
    under = df.sort_values("error").head(n_examples)

    # Largest overestimates (prediction >> true)
    over = df.sort_values("error", ascending=False).head(n_examples)

    # Best predictions (lowest absolute error)
    best = df.sort_values("abs_error").head(n_examples)

    under["category"] = "underestimate"
    over["category"] = "overestimate"
    best["category"] = "best"

    results = pd.concat([under, over, best], ignore_index=True)

    output_dir = os.path.join(model_dir, "example_imgs")
    os.makedirs(output_dir, exist_ok=True)

    # Copy images
    for _, row in results.iterrows():
        src = os.path.join(dataset_dir, row["image_filename"])

        name = f'{row["category"]}_{os.path.basename(row["image_filename"])}'
        dst = os.path.join(output_dir, name)

        shutil.copy(src, dst)

        row["copied_filename"] = name

    # Save CSV
    csv_path = os.path.join(output_dir, "examples.csv")
    results.to_csv(csv_path, index=False)

    print("Saved results to:", csv_path)
    print("Images copied to:", output_dir)
