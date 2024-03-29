from trainer import fit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import transforms
import os
from datasets import TripletAntsDataset
from networks import TripletNet, EmbeddingNet, FaceNet, EN2, TransferYOLO, EN3, CNN_EN, EN4, EN5, EN6, AntEmbeddingNet
from trainer import fit_triplet
from tester import test_model
import shutil
from plot_loss import plot_loss
from trainer import EarlyStopper
import pandas as pd
import matplotlib.pyplot as plt
import random
"""
This script trains a triplet network on the ant face dataset.
"""

######### PARAMETERS #########
embedding_network = EN4()
batch_size = 200
loss_margin = 1
loss_p = 1
learn_rate = 0.0005
gpu_id = "cuda:0"
gpu_parallel = True
scheduler_step_size = 7
scheduler_gamma = 0.1
num_epochs = 100
early_stopper = EarlyStopper(patience=7, min_delta=0.0)
data_dir = '../clean_ant_data'
csv_file = '../clean_ant_data/labels.csv'
##############################


# Enable benchmarking for faster runtime
cudnn.benchmark = True

# Resize and normalize the images
data_transforms = transforms.Compose([
                # transforms.Resize(375),
                # transforms.CenterCrop(375),
                transforms.Resize((512, 152)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


# Create dataset
dataset = TripletAntsDataset(csv_file, data_dir, transform=data_transforms)

# Set random seed for reproducibility of split among different scripts
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Split dataset into train, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])


# Create dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}


# Create model, loss function, optimizer, and scheduler
model = TripletNet(embedding_network)
loss_func = nn.TripletMarginLoss(margin=loss_margin, p=loss_p)
optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay = 0.0001)
device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")

if gpu_parallel:
    model = nn.DataParallel(model, device_ids=[0,1]) # Use both GPUs

scheduler = lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)


# Train model
final_model, final_loss = fit_triplet(model, dataloaders, loss_func, optimizer, scheduler, device, num_epochs=num_epochs, early_stopper=early_stopper)


# Check other models in models directory to see what the next model number should be
existing_models = os.listdir("../models")
latest_model_seq = 0
for existing_model in existing_models:
    tmp_id = existing_model.split("_")[-1]
    print("TMP ID: ", tmp_id)
    if int(tmp_id) >= latest_model_seq:
        latest_model_seq = int(tmp_id) + 1


# Create final model name
final_model_folder = "../models/triplet_net_body_" + str(latest_model_seq)
final_model_path = final_model_folder + "/best_model.pt"


# Save model
os.makedirs("../models", exist_ok=True)
os.makedirs(final_model_folder)
torch.save(final_model.state_dict(), final_model_path)


# Copy the loss to folder
path_to_loss = "../loss.csv" # Loss gets overwritten each time, so this is fine
shutil.copy(path_to_loss, final_model_folder)
plot_loss(final_model_folder)