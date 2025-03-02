"""
Contains functionality for creating Pytorch DataLoader's for Image classification data
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int=NUM_WORKERS
):
    """Creates training and testing DataLoaders.
    
    Takes in a training directory and testing directory path and turns them into
    PyTorch Datasets & then into PyTorch DataLoaders.
    
    Args:
        train_dir: Path to train data directory.
        test_dir: Path to test data directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of sample per batch in each of the DataLoaders.
        num_workers: An integar for number of workers per DataLoader.
    
    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
        Where class_names is a list of the target classes.
    
    Examples:
        train_dataloader, test_dataloader, class_names = create_dataloaders(train_dir="path/to/train_dir",
            test_dir="path/to/test_dir",
            transform=some_transform,
            batch_size=32,
            num_workers=4)
    """

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(root=train_dir, transform=transform, target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform, target_transform=None)

    # Get the class names
    class_names = train_data.classes

    # Turn datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_dataloader, test_dataloader, class_names
