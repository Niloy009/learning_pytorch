"""
Trains a PyTorch image classification model using device agnostic code.
"""

import os
import argparse
import torch
from torchvision import transforms
from torchmetrics import Accuracy
from timeit import default_timer as timer

import data_setup, engine, model_builder, utils


# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters...")


# Get an arg for num_epochs
parser.add_argument('--num_epochs', 
                    default=10, 
                    type=int, 
                    help="the number of epoch to train for.")

# Get an arg for batch_size
parser.add_argument("--batch_size", 
                    default=32, 
                    type=int, 
                    help="number of samples per batch")

# Get an arg for hidden_units
parser.add_argument("--hidden_units", 
                    default=10, 
                    type=int, 
                    help="number of hidden units between layers")

# Get an arg for learning rate
parser.add_argument("--learning_rate", 
                    default=0.001, 
                    type=float, 
                    help="learning rate to use for model")

# Get our arguments from the parser
args = parser.parse_args()

NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
print(f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units and a learning rate of {LEARNING_RATE}")


# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create tranform
data_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

# Create DataLoaders with help of data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir, 
                                                                               test_dir=test_dir, 
                                                                               transform=data_transform, 
                                                                               batch_size=BATCH_SIZE, 
                                                                               num_workers=os.cpu_count())

# Create model with help of mode_builder.py
model = model_builder.TinyVGG(input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)).to(device)

# Set Loss function, Optimizer, Accuracy
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
accuracy = Accuracy(task='multiclass', num_classes=len(class_names)).to(device)


# Start the timer
start_time = timer()

# Start training with help of engine.py
engine.train(model=model, 
             train_dataloader=train_dataloader, 
             test_dataloader=test_dataloader, 
             loss_fn=loss_fn, 
             optimizer=optimizer, 
             accuracy=accuracy, 
             epochs=NUM_EPOCHS, 
             device=device)

# End timer
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")


# Save the model with help of utils.py
utils.save_model(model=model, target_dir="models", model_name="tiny_vgg_model_v1.pth")
