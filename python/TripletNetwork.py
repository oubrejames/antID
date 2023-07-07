from trainer import fit
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
from datasets import TripletAntsDataset
from networks import TripletNet, EmbeddingNet
from trainer import fit_triplet

cudnn.benchmark = True

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(375),
        transforms.CenterCrop(375),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(375),
        transforms.CenterCrop(375),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(375),
        transforms.CenterCrop(375),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = '../ant_face_data'
csv_file = '../ant_face_data/labels.csv'
dataset = TripletAntsDataset(csv_file, data_dir)
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.1), int(len(dataset)*0.1)])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=20, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=20, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=20, shuffle=True, num_workers=4)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

model = TripletNet(EmbeddingNet())
criteria = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
fit_triplet(model, dataloaders, criteria, optimizer, scheduler, device, num_epochs=50)
