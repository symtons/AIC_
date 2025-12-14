# download_dataset.py
import os
import urllib.request
import zipfile

print("Downloading Flickr8k dataset...")

# Create directories
os.makedirs('data/Flickr8k', exist_ok=True)

# Download (from alternative sources)
dataset_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
captions_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

print("Downloading images...")
urllib.request.urlretrieve(dataset_url, "Flickr8k_Dataset.zip")

print("Downloading captions...")
urllib.request.urlretrieve(captions_url, "Flickr8k_text.zip")

print("Extracting...")
with zipfile.ZipFile("Flickr8k_Dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("data/Flickr8k")

with zipfile.ZipFile("Flickr8k_text.zip", 'r') as zip_ref:
    zip_ref.extractall("data/Flickr8k")

# Rename captions file
os.rename("data/Flickr8k/Flickr8k.token.txt", "data/Flickr8k/captions.txt")

print("✓ Dataset ready!")
