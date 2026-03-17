import json
import os
import matplotlib.pyplot as plt


def plot_training(model_dir):
    """
    Uses Matplotlib to display training and
    validation loss history, and save
    the plot next to the json.
    """
    json_path = os.path.join(model_dir, "training_history.json")

    # Load the history dictionary
    with open(json_path, "r") as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    epochs = range(1, len(train_loss) + 1)

    # Create plot
    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    # Save the figure
    output_path = os.path.join(model_dir, "training_history.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Show in Google Colab
    plt.show()
