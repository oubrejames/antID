import torch
import numpy as np
import time
import os
from tempfile import TemporaryDirectory
import csv

# EarlyStopper copied from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    """Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """Early stops the training if validation loss doesn't improve after a given patience.

        Args:
            validation_loss (float): validation loss.

        Returns:
            bool: Whether the training should be stopped or not
        """
    
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_one_epoch(model, data_loader, optimizer, criterion, device):
    """Train one epoch of the model (not used for triplet network).

    Args:
        model (nn.Module): ant face recognition model
        data_loader (torch.utils.data.DataLoader): train dataloader
        optimizer (optimizer): Choice of optimizer
        criterion (loss function): Choice of loss function
        device (torch.device): GPU or CPU

    Returns:
        Updated model, epoch loss, and epoch accuracy
    """

    # Get dataset size
    dataset_size = len(data_loader.dataset)

    # Set model to training mode
    model.train()

    # Initialize running count for loss and correct predictions
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # Zero gradients for the optimizer
        optimizer.zero_grad()

        # Enable gradient tracking if only in train
        torch.set_grad_enabled(True)

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Backward pass and optimize if in training phase
        loss.backward()
        optimizer.step()

        # Calculate loss and total correct predictions
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    return model, epoch_loss, epoch_acc

def validate_one_epoch(model, data_loader, optimizer, criterion, device):
    """Validate one epoch of the model (not used for triplet network).

    Args:
        model (nn.Module): ant face recognition model
        data_loader (torch.utils.data.DataLoader): train dataloader
        optimizer (optimizer): Choice of optimizer
        criterion (loss function): Choice of loss function
        device (torch.device): GPU or CPU

    Returns:
        Epoch loss and epoch accuracy
    """

    # Get dataset size
    dataset_size = len(data_loader.dataset)

    # Set model to evaluation mode
    model.eval()

    # Initialize running count for loss and correct predictions
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero gradients for the optimizer
        optimizer.zero_grad()

        # Enable gradient tracking if only in train
        torch.set_grad_enabled(False)

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Calculate loss and total correct predictions
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    return epoch_loss, epoch_acc

def fit(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=50, early_stopper=None):
    """Fit the model to the data for a given number of epochs (not used for triplet network).

    Args:
        model (nn.Module): ant face recognition model
        dataloaders (list): list of train, validation, and test dataloaders
        criterion (loss function): Choice of loss function
        optimizer (optimizer): Choice of optimizer
        scheduler (lr_scheduler.StepLR): Learning rate scheduler
        device (torch.device): GPU or CPU
        num_epochs (int, optional): Amount of epochs to train over. Defaults to 50.

    Returns:
        Trained model
    """

    # Start measuring time of training
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        # Save model parameters
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)

        # Initialize best validation accuracy
        best_val_acc = 0.0

        model.to(device)
        # Iterate over epochs
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Train and validate
            model, train_loss, train_acc = train_one_epoch(model, dataloaders['train'], optimizer, criterion, device)
            scheduler.step() # Update learning rate
            val_loss, val_acc = validate_one_epoch(model, dataloaders['val'], optimizer, criterion, device)
            print('Training Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))
            print('Validation Loss: {:.4f} Acc: {:.4f}'.format(val_loss, val_acc))

            # Update best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_params_path)

            # Check if early stopper should stop training
            if early_stopper is not None:
                if early_stopper.early_stop(val_loss):
                    print("Stopping early. Validation loss did not improve for {} epochs.".format(early_stopper.patience))
                    break

                # Print early stopper count to see how its tracking
                print("Early stopper count: ", early_stopper.counter)
                print('\n')

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_val_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model

def train_one_epoch_triplet(model, data_loader, optimizer, criterion, device):
    """Train one epoch of the model (used for triplet network).

    Args:
        model (nn.Module): ant face recognition model
        data_loader (torch.utils.data.DataLoader): train dataloader
        optimizer (optimizer): Choice of optimizer
        criterion (loss function): Choice of loss function
        device (torch.device): GPU or CPU

    Returns:
        Average loss for the epoch
    """

    # Initialize running count for loss
    running_loss = 0.0

    # Iterate over data.
    for anchor, positive, negative, label in data_loader:
        # Send data to device
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # Zero gradients for the optimizer
        optimizer.zero_grad()

        # Forward pass
        # Get model embeddings
        anchor_output, positive_output, negative_output = model(anchor, positive, negative)

        # Compute triplet loss
        loss = criterion(anchor_output, positive_output, negative_output)

        # Backward pass and optimize if in training phase
        loss.backward()
        optimizer.step()
        
        running_loss += float(loss.item())
    
    return running_loss / len(data_loader.dataset)

def validate_one_epoch_triplet(model, data_loader, optimizer, criterion, device):
    """Validate one epoch of the model (used for triplet network).

    Args:
        model (nn.Module): ant face recognition model
        data_loader (torch.utils.data.DataLoader): train dataloader
        optimizer (optimizer): Choice of optimizer
        criterion (loss function): Choice of loss function
        device (torch.device): GPU or CPU

    Returns:
        Average loss for the epoch
    """
    running_loss = 0.0

    # Iterate over data.
    for anchor, positive, negative, label in data_loader:
        # Send data to device
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # Zero gradients for the optimizer
        optimizer.zero_grad()

        # Forward pass
        # Get model embeddings
        anchor_output, positive_output, negative_output = model(anchor, positive, negative)

        # Compute triplet loss
        loss = criterion(anchor_output, positive_output, negative_output)

        running_loss += float(loss.item())

    return running_loss / len(data_loader.dataset)

def fit_triplet(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=50, early_stopper=None):
    """Fit the model to the data for a given number of epochs (used for triplet network).

    Args:
        model (nn.Module): ant face recognition model
        dataloaders (list): list of train, validation, and test dataloaders
        criterion (loss function): Choice of loss function
        optimizer (optimizer): Choice of optimizer
        scheduler (lr_scheduler.StepLR): Learning rate scheduler
        device (torch.device): GPU or CPU
        num_epochs (int, optional): Amount of epochs to train over. Defaults to 50.

    Returns:
        Trained model
    """

    # Start measuring time of training
    since = time.time()

    # Save loss as a csv (will overwrite previous training loss csv)
    with open(os.path.join("../", "loss.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(['val_loss', 'train_loss'])

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        # Save model parameters and move to device
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        model = model.to(device)
        torch.save(model.state_dict(), best_model_params_path)

        # Initialize best validation loss
        best_val_loss = 9999.0

        # Iterate over epochs
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Set model to training mode and train one epoch
            model.train()
            train_loss = train_one_epoch_triplet(model, dataloaders['train'], optimizer, criterion, device)
            scheduler.step() # Update learning rate

            # Evaluate model on validation set 
            model.eval()
            val_loss = validate_one_epoch_triplet(model, dataloaders['val'], optimizer, criterion, device)

            print('Training Loss: {:.4f}'.format(train_loss))
            print('Validation Loss: {:.4f}'.format(val_loss))

            # Save loss as a csv
            with open(os.path.join("../", "loss.csv"), "a") as f:
                writer = csv.writer(f)
                writer.writerow([val_loss, train_loss])

            # Save model if validation loss is lower than previous best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            # Check if early stopper should stop training
            if early_stopper is not None:
                if early_stopper.early_stop(val_loss):
                    print("Stopping early. Validation loss did not improve for {} epochs.".format(early_stopper.patience))
                    break

                # Print early stopper count to see how its tracking
                print("Early stopper count: ", early_stopper.counter)
                print('\n')

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model, best_val_loss
