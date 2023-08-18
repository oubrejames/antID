from trainer import fit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import transforms
import os
from datasets import TripletAntsDataset, AntsDataset
from networks import TripletNet, EmbeddingNet, FaceNet, EN2, TransferYOLO, CNN
from trainer import fit_triplet
from tester import test_model
import shutil
from plot_loss import plot_loss
from trainer import EarlyStopper
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets
from tester import test_classifier

"""
This script trains a CNN calssifier to identitify ants within a closed set.
"""

######### PARAMETERS #########
batch_size = 100
learn_rate = 0.01
gpu_id = "cuda:0"
num_epochs = 200
early_stopper = EarlyStopper(patience=7, min_delta=0.001)
##############################


# Enable benchmarking for faster runtime
cudnn.benchmark = True


# Resize and normalize the images
data_transforms = transforms.Compose([
                transforms.Resize(375),
                transforms.CenterCrop(375),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


# Create dataset
data_dir = '../body_folder_dataset'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms)
                  for x in ['train', 'val', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

class_names = image_datasets['train'].classes

device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")


# Instatiate model
model = CNN(len(class_names))
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.7)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# Train model
final_model = fit(model, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, device, num_epochs=num_epochs, early_stopper=early_stopper)


# Check other models in models directory to see what the next model number should be
existing_models = os.listdir("../models")
latest_model_seq = 0
for existing_model in existing_models:
    tmp_id = existing_model.split("_")[-1]
    print("TMP ID: ", tmp_id)
    if int(tmp_id) >= latest_model_seq:
        latest_model_seq = int(tmp_id) + 1


# Create final model name
final_model_folder = "../models/cnn_classifier_" + str(latest_model_seq)
final_model_path = final_model_folder + "/best_model.pt"


# Save model
os.makedirs("../models", exist_ok=True)
os.makedirs(final_model_folder)
torch.save(final_model.state_dict(), final_model_path)


# Copy the loss to folder
path_to_loss = "../loss.csv" # Loss gets overwritten each time, so this is fine
shutil.copy(path_to_loss, final_model_folder)
plot_loss(final_model_folder)


# Test the model
model_test_acc = test_classifier(final_model, dataloaders['test'], device)
print("Final accuracy on test set: ", model_test_acc)