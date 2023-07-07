import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import csv

path_to_labeled_images = "../labeled_images"
path_to_dataset = "../ant_face_data"
os.makedirs(path_to_dataset, exist_ok=True)

train_dir = os.path.join(path_to_dataset, 'train')
test_dir = os.path.join(path_to_dataset, 'test')
val_dir = os.path.join(path_to_dataset, 'val')
test_size = 0.2  # Test set size
val_size = 0.25  # Validation set size (25% of the remaining data after the test split)
random_state = 42

# Create the train, test, and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Create csv file with image paths and labels
head = ['label', 'img_path']

with open(os.path.join(path_to_dataset, "labels.csv"), "a") as f:
    writer = csv.writer(f)
    writer.writerow(head)

# Split the dataset into train, test, and validation sets
labels = os.listdir(path_to_labeled_images)
for label in labels:
    label_dir = os.path.join(path_to_labeled_images, label)
    train_label_dir = os.path.join(train_dir, label)
    test_label_dir = os.path.join(test_dir, label)
    val_label_dir = os.path.join(val_dir, label)
    dst_label_dir = os.path.join(path_to_dataset, label)

    image_files = os.listdir(label_dir)
    
    # Write image paths and labels to csv file
    for image_file in image_files:
        row = [label, image_file]
        with open(os.path.join(path_to_dataset, "labels.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Save images to labeled dataset
        src_path = os.path.join(label_dir, image_file)
        # dst_path = os.path.join(dst_label_dir, image_file)
        os.makedirs(dst_label_dir, exist_ok=True)
        shutil.copy(src_path, dst_label_dir)

    # train_files, test_val_files = train_test_split(image_files, test_size=test_size, random_state=random_state)
    # test_files, val_files = train_test_split(test_val_files, test_size=val_size/(1-test_size), random_state=random_state)

    # for file in train_files:
    #     src_path = os.path.join(label_dir, file)
    #     dst_path = os.path.join(train_label_dir, file)
    #     shutil.copy(src_path, dst_path)

    # for file in test_files:
    #     src_path = os.path.join(label_dir, file)
    #     dst_path = os.path.join(test_label_dir, file)
    #     shutil.copy(src_path, dst_path)

    # for file in val_files:
    #     src_path = os.path.join(label_dir, file)
    #     dst_path = os.path.join(val_label_dir, file)
    #     shutil.copy(src_path, dst_path)
