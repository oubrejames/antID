import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import csv

"""
Given a path to a folder containing images, create a csv file containing the image paths and labels.
"""

########## Parameters ##########
"""
Params here are to choose what images you want to create your dataset. I had 4 sets of images to 
work with: ant faces to train and validate with, unseen ants purely for testing, full ants to train
and validate with, unseen ant bodies purely for testing.
"""

dataset_idx = 3 # Change dataset idx to pick what data you want to format
all_datasets = ['ant_face_data', 'unseen_data', 'ant_body_data', 'unseen_body_data']
dataset_to_format = all_datasets[dataset_idx]

image_folders = ['labeled_images', 'unseen_images', 'labeled_images_bodies', 'unseen_body_imgs']
image_folder_to_use = image_folders[dataset_idx]

path_to_labeled_images = "../" + 
path_to_dataset = "../" + dataset_to_format
################################

# Create dataset folder
os.makedirs(path_to_dataset, exist_ok=True)

# Loop through each label folder
labels = os.listdir(path_to_labeled_images)
for label in labels:
    label_dir = os.path.join(path_to_labeled_images, label)
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
        os.makedirs(dst_label_dir, exist_ok=True)
        shutil.copy(src_path, dst_label_dir)