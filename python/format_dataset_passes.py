import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
import csv

"""
Given a path to a folder containing images, create a csv file containing the image paths and labels.
"""

########## Parameters ##########
dataset_to_format = 'test_pass_dataset'

image_folder_to_use = "unseen_body_passes/"

path_to_labeled_images = "../" + image_folder_to_use
path_to_dataset = "../" + dataset_to_format
################################

# Create dataset folder
os.makedirs(path_to_dataset, exist_ok=True)

# Loop through each label folder
ant_folders = os.listdir(path_to_labeled_images)
for ant_id in ant_folders:
    ant_id_dir = os.path.join(path_to_labeled_images, ant_id)
    dst_ant_id_dir = os.path.join(path_to_dataset, ant_id)

    for single_pass in os.listdir(ant_id_dir):
        single_pass_dir = os.path.join(ant_id_dir, single_pass)

        image_files = os.listdir(single_pass_dir)
        pass_number = single_pass.split('/')[-1]
        dst_pass_dir = os.path.join(dst_ant_id_dir, single_pass)

        # Write image paths and ant_ids to csv file
        for image_file in image_files:
            row = [ant_id, image_file, pass_number]
            with open(os.path.join(path_to_dataset, "labels.csv"), "a") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            # Save images to labeled dataset
            src_path = os.path.join(single_pass_dir, image_file)
            os.makedirs(dst_pass_dir, exist_ok=True)
            shutil.copy(src_path, dst_pass_dir)