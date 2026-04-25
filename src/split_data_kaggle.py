import os
import shutil
import random

# Paths
original_dataset_dir = "/Users/asibulislam/Downloads/PetImages"
base_dir = "../data/cats_dogs"

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

train_cats = os.path.join(train_dir, "cats")
train_dogs = os.path.join(train_dir, "dogs")
val_cats = os.path.join(val_dir, "cats")
val_dogs = os.path.join(val_dir, "dogs")

# Create directories
for folder in [train_cats, train_dogs, val_cats, val_dogs]:
    os.makedirs(folder, exist_ok=True)

# Function to split data
def split_data(src_dir, train_dir, val_dir, split=0.8):
    files = os.listdir(src_dir)

    # Remove corrupted files (very important for Kaggle dataset)
    files = [f for f in files if f.endswith(".jpg")]

    random.shuffle(files)

    split_index = int(len(files) * split)

    train_files = files[:split_index]
    val_files = files[split_index:]

    for f in train_files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(train_dir, f))

    for f in val_files:
        shutil.copy(os.path.join(src_dir, f), os.path.join(val_dir, f))


# Split cats
split_data(
    os.path.join(original_dataset_dir, "Cat"),
    train_cats,
    val_cats
)

# Split dogs
split_data(
    os.path.join(original_dataset_dir, "Dog"),
    train_dogs,
    val_dogs
)

print("Dataset split complete!")