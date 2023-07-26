import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import random
from PIL import Image

class TripletAntsDataset(Dataset):
    """
    Dataset class for the triplet ants dataset. Returns a triplet of images and the label of the 
    anchor image.
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if isinstance(csv_file, pd.DataFrame):
            self.labels = csv_file
        elif isinstance(csv_file, str):
            self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anchor_path = os.path.join(self.root_dir,               # Directory to all ant folders
                                self.labels.iloc[idx, 0],       # Directory to specific ant folder
                                self.labels.iloc[idx, 1])       # Filename of anchor image

        anchor_image = Image.open(anchor_path)

        positive_path = anchor_path
        while positive_path == anchor_path:
            pos_img_name = random.choice(os.listdir(os.path.join(self.root_dir, self.labels.iloc[idx, 0])))
            positive_path = os.path.join(self.root_dir, 
                                         self.labels.iloc[idx, 0], 
                                         pos_img_name)

        positive_image = Image.open(positive_path)
    
        positive_dir=  os.path.join(self.root_dir,               # Directory to all ant folders
                                self.labels.iloc[idx, 0])       # Directory to specific ant folder
        negative_dir = positive_dir

        while negative_dir == positive_dir:
            negative_dir = random.choice(os.listdir(self.root_dir))

            negative_dir_path = os.path.join(self.root_dir, negative_dir)
            if os.path.isfile(negative_dir_path):
                negative_dir = positive_dir
                continue

            neg_img_name = random.choice(os.listdir(negative_dir_path))
            negative_path = os.path.join(negative_dir_path, neg_img_name)

        negative_image = Image.open(negative_path)

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image, str(self.labels.iloc[idx, 0])

class AntsDataset(Dataset):
    """
    Dataset class for the ants dataset. Returns an image and the label of the image.
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if isinstance(csv_file, pd.DataFrame):
            self.labels = csv_file
        elif isinstance(csv_file, str):
            self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anchor_path = os.path.join(self.root_dir,               # Directory to all ant folders
                                self.labels.iloc[idx, 0],       # Directory to specific ant folder
                                self.labels.iloc[idx, 1])       # Filename of anchor image

        anchor_image = Image.open(anchor_path)

        if self.transform:
            anchor_image = self.transform(anchor_image)

        # Get ant id
        ant_id = torch.tensor(int(self.labels.iloc[idx, 0].split("_")[-1]))
        return anchor_image, ant_id
