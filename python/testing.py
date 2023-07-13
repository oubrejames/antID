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
import csv

def test_model(model, test_loader, device):
    model = model.to(device)
    for anchor, positive, negative, label in test_loader:
        # anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        anchor_output, positive_output, negative_output = model(anchor, positive, negative)
        
        # Get L2 norm of each output
        anchor_norm = np.linalg.norm(anchor_output)
        positive_norm = np.linalg.norm(positive_output)
        negative_norm = np.linalg.norm(negative_output)
        
        # Calculate the squared L2 distance between each output
        anchor_positive_dist = np.linalg.norm(anchor_output - positive_output)**2
        anchor_negative_dist = np.linalg.norm(anchor_output - negative_output)**2
        
        print("Positive distance: ", anchor_positive_dist)
        print("Negative distance: ", anchor_negative_dist)

    return None

# def main():
#     # Load model
#     model = torch.load('../bestloss0.pt')

    
# if __name__ == '__main__':
#     main()