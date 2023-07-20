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

# From https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_one_epoch(model, data_loader, optimizer, criterion, device):

    dataset_size = len(data_loader.dataset)


    model.train()  # Set model to training mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        torch.set_grad_enabled(True)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    return model, epoch_loss, epoch_acc

def validate_one_epoch(model, data_loader, optimizer, criterion, device):
    dataset_size = len(data_loader.dataset)

    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        torch.set_grad_enabled(False)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    return epoch_loss, epoch_acc

def fit(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=50):
    # Start measuring time of training
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_loss = 0.0
        early_stopper = EarlyStopper(patience=15, min_delta=0.001)
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            model, train_loss, train_acc = train_one_epoch(model, dataloaders['train'], optimizer, criterion, device)
            scheduler.step() # Update learning rate

            val_loss, val_acc = validate_one_epoch(model, dataloaders['val'], optimizer, criterion, device)
            print('Training Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))
            print('Validation Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))

            if val_acc > best_loss:
                best_loss = val_acc
                torch.save(model.state_dict(), best_model_params_path)

            if early_stopper.early_stop(val_loss):
                print("Stopping early. Validation loss did not improve for {} epochs.".format(early_stopper.patience))
                break

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_loss:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

def train_one_epoch_triplet(model, data_loader, optimizer, criterion, device):
    running_loss = 0.0
    
    for anchor, positive, negative, label in data_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        anchor_output, positive_output, negative_output = model(anchor, positive, negative)
        loss = criterion(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()
        
        running_loss += float(loss.item())
    
    return float(loss.item()) / len(data_loader.dataset)

def validate_one_epoch_triplet(model, data_loader, optimizer, criterion, device):
    running_loss = 0.0
    
    for anchor, positive, negative, label in data_loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        
        optimizer.zero_grad()
        anchor_output, positive_output, negative_output = model(anchor, positive, negative)
        loss = criterion(anchor_output, positive_output, negative_output)
        
        running_loss += float(loss.item())
    
    return float(loss.item()) / len(data_loader.dataset)

def fit_triplet(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=50):
    # Start measuring time of training
    since = time.time()

    # Save loss as a csv
    with open(os.path.join("../", "loss.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(['val_loss', 'train_loss'])

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        model = model.to(device)
        torch.save(model.state_dict(), best_model_params_path)
        best_loss = 9999.0
        
        
        early_stopper = EarlyStopper(patience=15, min_delta=0.001/1000)
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            model.train()
            train_loss = train_one_epoch_triplet(model, dataloaders['train'], optimizer, criterion, device)
            scheduler.step() # Update learning rate

            model.eval()
            val_loss = validate_one_epoch_triplet(model, dataloaders['val'], optimizer, criterion, device)
            print('Training Loss (1000x): {:.4f}'.format(train_loss*1000))
            print('Validation Loss (1000X): {:.4f}'.format(val_loss*1000))
            print('\n')

            # Save loss as a csv
            with open(os.path.join("../", "loss.csv"), "a") as f:
                writer = csv.writer(f)
                writer.writerow([val_loss, train_loss])

            # deep copy the model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            if early_stopper.early_stop(val_loss):
                print("Stopping early. Validation loss did not improve for {} epochs.".format(early_stopper.patience))
                break

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        # print(f'Best val Acc: {best_loss:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model, best_loss
