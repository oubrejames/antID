import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import random
from PIL import Image
from torchvision import transforms

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

        # Get ant id number
        ant_id = torch.tensor(int(self.labels.iloc[idx, 0].split("_")[-1]))
        return anchor_image, (self.labels.iloc[idx, 0])

class TripletPasses(Dataset):
    """
    Returns a triplet consisting of one anchor image and a tensor of images that all correspond to 
    an ant doing one walk under the camera for both the negative and positive of the triplet.
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

        ant_name = self.labels.iloc[idx, 0]
        anchor_img_name = self.labels.iloc[idx, 1]
        anchor_pass = self.labels.iloc[idx, 2]
        anchor_pass_number = int(self.labels.iloc[idx, 2].split('_')[-1])

        anchor_ant_path = os.path.join(self.root_dir,               # Directory to all ant folders
                                ant_name)       # Directory to specific ant folder

        anchor_img_path = os.path.join(self.root_dir,               # Directory to all ant folders
                                ant_name,       # Directory to specific ant folder
                                anchor_pass,
                                anchor_img_name)       # Filename of anchor image

        anchor_pass_path = os.path.join(self.root_dir,               # Directory to all ant folders
                                ant_name,       # Directory to specific ant folder
                                anchor_pass)

        anchor_image = Image.open(anchor_img_path)

        positive_pass_path = anchor_pass_path
        positive_pass_number = anchor_pass_number

        while positive_pass_number == anchor_pass_number: # Run until you get image in a different pass
            # Choose a random pass folder matching anchor id
            positive_pass = random.choice(os.listdir(anchor_ant_path))
            positive_pass_number = int(positive_pass.split('_')[-1])
            positive_pass_path = os.path.join(self.root_dir,            # Images dir
                                         ant_name,      # Antid dir
                                         "pass_"+str(positive_pass_number)) # Particular pass dir


        # Get all images in the positive pass dir
        positive_images = []
        for pos_im in os.listdir(positive_pass_path):
            pos_im_path = os.path.join(positive_pass_path, pos_im)
            positive_images.append(Image.open(pos_im_path))

        positive_dir= anchor_ant_path
        negative_dir = positive_dir

        while negative_dir == positive_dir:
            negative_dir = random.choice(os.listdir(self.root_dir))

            negative_dir_path = os.path.join(self.root_dir, negative_dir)
            if os.path.isfile(negative_dir_path): # make sure its not labels.csv
                negative_dir = positive_dir
                continue

            # Choose random negative pass
            negative_pass = random.choice(os.listdir(negative_dir_path))
            negative_pass_path = os.path.join(negative_dir_path, negative_pass)


        negative_images = []
        for neg_im in os.listdir(negative_pass_path):
            neg_im_path = os.path.join(negative_pass_path, neg_im)
            negative_images.append(Image.open(neg_im_path))


        if self.transform:
            anchor_image = self.transform(anchor_image)
            transformed_pos_images = [self.transform(img) for img in positive_images]
            pos_ims_tensor = torch.stack(transformed_pos_images)

            transformed_neg_images = [self.transform(img) for img in negative_images]
            neg_ims_tensor = torch.stack(transformed_neg_images)

        return anchor_image, pos_ims_tensor, neg_ims_tensor, str(ant_name)