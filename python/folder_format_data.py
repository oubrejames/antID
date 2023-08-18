import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split

######### PARAMETERS #########
ant_face_flag = False # Choose if using ant face or body data
##############################

if ant_face_flag:
    path_to_labeled_images = "../labeled_images"
    path_to_dataset = "../folder_dataset"
else:
    path_to_labeled_images = "../labeled_images_bodies"
    path_to_dataset = "../body_folder_dataset"

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

# Split the dataset into train, test, and validation sets
labels = os.listdir(path_to_labeled_images)
for label in labels: # Loop through each ant folder in labeled images
    label_dir = os.path.join(path_to_labeled_images, label)
    train_label_dir = os.path.join(train_dir, label)
    test_label_dir = os.path.join(test_dir, label)
    val_label_dir = os.path.join(val_dir, label)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    image_files = os.listdir(label_dir) # Get all the images of a particular ant
    print("Number of images for ", label, ": ", len(image_files))

    train_files, test_val_files = train_test_split(image_files, test_size=test_size, random_state=random_state)
    test_files, val_files = train_test_split(test_val_files, test_size=val_size, random_state=random_state)

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