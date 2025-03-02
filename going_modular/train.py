"""
Trains a PyTorch image classification model using device agnostic code.
"""

import os
import torch
from torchvision import transforms
from torchmetrics import Accuracy
from timeit import default_timer as timer

import data_setup, engine, model_builder, utils


NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001


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
utils.save_model(model=model, target_dir="models", model_name="tiny_vgg_model_v0.pth")
