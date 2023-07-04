import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
from matplotlib import pyplot as plt

# Ignore warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

# Read the csv file
face_df = pd.read_csv('../ant_faces_dataset/faces/labels.csv')
print(face_df.head())
# Read a single image (test)
img_name = face_df.iloc[0, 0]
print('Image name: {}'.format(img_name))

# Dataset class for ant faces
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class AntFaceDataset(Dataset):
    """Ant Face dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ant_face_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ant_face_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.ant_face_df.iloc[idx, 1])
        image = io.imread(img_name)
        id = self.ant_face_df.iloc[idx, 0]
        id = id[-1] # Get the last character of the id (the number)
        sample = {'id':id, 'image': image}

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, id = sample['image'], sample['id']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'id': id}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, id = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'id': id}

face_dataset = AntFaceDataset(csv_file='../ant_faces_dataset/faces/labels.csv',
                                    root_dir='../ant_faces_dataset/faces/',
                                    transform=transforms.Compose([
                                        Rescale(375),
                                        ToTensor()
                                    ]))

dataloader = DataLoader(face_dataset, batch_size=4,
                        shuffle=True, num_workers=0)