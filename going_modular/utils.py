"""
Contains various utility functions for PyTorch model training and saving.
"""

import os
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
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


def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str = None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Create a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir
    
    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra

    Where timestamp is current date in YYYY-MM-DD format

    Args:
        experiment_name (str): Name of the experiment.
        model_name (str): Name of the model
        extra (str, optional): Anything extra to add to the directory.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to the specific log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    # Get timestamp of current date in reverse order (YYYY-MM-DD)
    timestamp = datetime.now().strftime("%Y-%b-%d")

    if extra:
        # create log directory path
        log_dir = os.path.join('runs', timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join('runs', timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter saving to {log_dir}")
    return SummaryWriter(log_dir=log_dir)
    




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
