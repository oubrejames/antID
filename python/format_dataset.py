"""This script loops through the folder of labeled ant heads and creates a csv file with the following columns:
"ant_id", "image_file_name"
"""


# Program outline:
# 1. Loop through all folders in labeled_images
# 2. Loop through all images in each folder
# 3. Create a row in the csv file for each image
# 4. Rename image (so no repeats) and move to the new folder
# 5. Save the csv file

# Output directory structure:
# data
#   faces
#       im0.jpg
#       im1.jpg
#       ...
#       labels.csv

# Imports
import os
import csv

# Loop through all folders in labeled_images (each folder contains images of one ant)
path_to_labeled_images = "../../tmp_data"
path_to_dataset = "../ant_faces_dataset"

for ant_dir in os.listdir(path_to_labeled_images):
    # Create path to ant folder
    ant_dir_path = os.path.join(path_to_labeled_images, ant_dir)

    # Loop through all images in each folder
    for img in os.listdir(ant_dir_path):
        # Create path to image
        img_path = os.path.join(ant_dir_path, img)

        # Create path to new image
        new_img = ant_dir + "_" + img.split(".")[0] + ".jpg"
        new_img_path = os.path.join(path_to_dataset, "faces", new_img)
        
        # Create row for csv file
        row = [ant_dir, new_img]

        # Rename image and move to new folder
        os.rename(img_path, new_img_path)

        # Save row to csv file
        with open(os.path.join(path_to_dataset, "labels.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow(row)