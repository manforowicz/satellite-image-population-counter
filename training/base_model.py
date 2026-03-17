import torch
import torch.nn as nn
from torchvision import models


def base_model(dropout_prob: float=0.0, train_body: bool=False):
    """
    Returns a pre-trained ResNet-50 model with
    the final layer modified.
    """
    # pretrained ResNet-50,
    # with final fully connected layer
    # modified to output a single number (log1p of population density)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_prob),
        nn.Linear(num_features, 1),
    )

    # freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze deeper layers if applicable
    if train_body:
        later_layers = [
            model.layer3,
            model.layer4,
            model.fc
        ]
        for layer in later_layers:
            for param in layer.parameters():
                param.requires_grad = True

    # Final FC is always unfrozen
    for param in model.fc.parameters():
        param.requires_grad = True

    return model
