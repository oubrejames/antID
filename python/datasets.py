import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from skimage import io, transform
import random

class TripletAntsDataset(Dataset):
    """"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
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

        anchor_image = io.imread(anchor_path)

        positive_path = anchor_path
        while positive_path == anchor_path:
            pos_img_name = random.choice(os.listdir(os.path.join(self.root_dir, self.labels.iloc[idx, 0])))
            positive_path = os.path.join(self.root_dir, 
                                         self.labels.iloc[idx, 0], 
                                         pos_img_name)

        positive_image = io.imread(positive_path)

        positive_dir=  os.path.join(self.root_dir,               # Directory to all ant folders
                                self.labels.iloc[idx, 0])       # Directory to specific ant folder
        negative_dir = positive_dir
        while negative_dir == positive_dir:
            negative_dir = random.choice(os.listdir(self.root_dir))
            neg_img_name = random.choice(os.listdir(os.path.join(self.root_dir, negative_dir)))
            negative_path = os.path.join(self.root_dir, negative_dir, neg_img_name)

        negative_image = io.imread(positive_path)

        # sample = {'anchor': anchor_image,
        #           'positive': positive_image,
        #           'negative': negative_image,
        #           'label': self.labels.iloc[idx, 0]}

        if self.transform:
            # sample = {'anchor':  self.transform(anchor_image),
            #           'positive':  self.transform(positive_image),
            #           'negative':  self.transform(negative_image),
            #           'label': self.labels.iloc[idx, 0]}
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image, str(self.labels.iloc[idx, 0])