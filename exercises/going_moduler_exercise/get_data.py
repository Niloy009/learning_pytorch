
# 1. get the data
import os
import requests
import zipfile
from pathlib import Path


URL = "https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip"
# Setup path to a data folder
data_path = Path('data/')
image_path = data_path / 'pizza_steak_sushi'


# If the imgae folder doesn't exist, download it and prepare it
if image_path.is_dir():
    print(f'{image_path} directory already exists... skipping download')
else:
    print(f'{image_path} directory does not exist... creating one....')
    image_path.mkdir(parents=True, exist_ok=True)


# Download pizza, steak, and sushi data
with open(data_path/ 'pizza_steak_sushi.zip', 'wb') as f:
    request = requests.get(URL)
    print("Downloading pizza, steak and sushi data...")
    f.write(request.content)
    print("Download done.......")


# Unzip pizza, steak and sushi data
with zipfile.ZipFile(data_path / 'pizza_steak_sushi.zip', 'r') as zip_ref:
    print("Unzipping pizza, steak and sushi data")
    zip_ref.extractall(image_path)
    print("Extracted All....")

# Remove zip file
os.remove(data_path / "pizza_steak_sushi.zip")
