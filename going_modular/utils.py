"""
Contains various utility functions for PyTorch model training and saving.
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


def save_model(model: torch.nn.Module, 
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory
    
    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include either ".pth" or ".pt" as the file extension.

    Example Usage:
        save_model(model=model_1, 
                   target_dir='models', 
                   model_name='tiny_vgg_model.pth')    
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pth' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_accuracy": [...],
             "test_loss": [...],
             "test_accuracy": [...]}
    """
    # Get the loss values from the results dictionary (training and testing)
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values from the results dictionary (training & testing)
    accuracy = results["train_accuracy"]
    test_accuracy = results["test_accuracy"]

    # Get the epochs
    epochs = range(len(results["train_loss"]))

    # Setup the plot
    plt.figure(figsize=(15,7))

    # plot the loss
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, test_loss, label="Test Loss")
    plt.title("Loss Curves")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()

    # plot the loss
    plt.subplot(1,2,2)
    plt.plot(epochs, accuracy, label="Train Accuracy")
    plt.plot(epochs, test_accuracy, label="Test Accuracy")
    plt.title("Accuracy Curves")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
