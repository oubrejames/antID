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
from testing import test_model, test_thresholds

cudnn.benchmark = True

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'crop_norm': transforms.Compose([
        transforms.Resize(375),
        transforms.CenterCrop(375),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = '../unseen_data'
csv_file = '../unseen_data/labels.csv'
dataset = TripletAntsDataset(csv_file, data_dir, transform=data_transforms['crop_norm'])

test_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)


data_dir = '../ant_face_data'
csv_file = '../ant_face_data/labels.csv'
full_dataset = TripletAntsDataset(csv_file, data_dir, transform=data_transforms['crop_norm'])
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
val_loader = torch.utils.data.DataLoader(val_set, batch_size=100, shuffle=True, num_workers=4)


model = TripletNet(EmbeddingNet())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model = nn.DataParallel(model, device_ids=[0, 1]) # Use both GPUs

model.load_state_dict(torch.load('../models/triplet_net_3/best_model.pt'))
model.eval()
latest_model_seq =3


# ##### Get thresh
# # Get model sequence number
# existing_models = os.listdir("../models")
# latest_model_seq = 0
# for existing_model in existing_models:
#     tmp_id = existing_model.split(".")[0][-1]
#     print("TMP ID: ", tmp_id)
#     if int(tmp_id) > latest_model_seq:
#         latest_model_seq = tmp_id


# Create final model name
final_model_folder = "../models/triplet_net_" + str(latest_model_seq)

# best_threshold, average_thresh = test_thresholds(model, test_loader, device, final_model_folder)
#220
average_thresh = 200
print("##### Average threshold : ", average_thresh)
test_model(model, test_loader, device, average_thresh)
