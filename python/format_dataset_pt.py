import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

path_to_labeled_images = "../labeled_images"
path_to_dataset = "../ant_face_data"

train_dir = os.path.join(save_dir, 'train')
test_dir = os.path.join(save_dir, 'test')
val_dir = os.path.join(save_dir, 'val')
test_size = 0.2  # Test set size
val_size = 0.25  # Validation set size (25% of the remaining data after the test split)
random_state = 42

# Create the train, test, and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Split the dataset into train, test, and validation sets
labels = os.listdir(dataset_dir)
for label in labels:
    label_dir = os.path.join(dataset_dir, label)
    train_label_dir = os.path.join(train_dir, label)
    test_label_dir = os.path.join(test_dir, label)
    val_label_dir = os.path.join(val_dir, label)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    image_files = os.listdir(label_dir)
    train_files, test_val_files = train_test_split(image_files, test_size=test_size, random_state=random_state)
    test_files, val_files = train_test_split(test_val_files, test_size=val_size/(1-test_size), random_state=random_state)

    for file in train_files:
        src_path = os.path.join(label_dir, file)
        dst_path = os.path.join(train_label_dir, file)
        shutil.copy(src_path, dst_path)

    for file in test_files:
        src_path = os.path.join(label_dir, file)
        dst_path = os.path.join(test_label_dir, file)
        shutil.copy(src_path, dst_path)

    for file in val_files:
        src_path = os.path.join(label_dir, file)
        dst_path = os.path.join(val_label_dir, file)
        shutil.copy(src_path, dst_path)
