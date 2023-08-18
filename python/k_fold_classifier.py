import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from networks import CNN
import torch.backends.cudnn as cudnn
from torchvision import datasets
from tester import test_classifier

######### PARAMETERS #########
batch_size = 100
loss_p = 2
learn_rate = 0.01
gpu_id = "cuda:1"
num_epochs = 2
##############################


# Enable benchmarking for faster runtime
cudnn.benchmark = True


# Configuration options
k_folds = 5
num_epochs = 25
loss_function = nn.CrossEntropyLoss()

# For fold results
results = {}

# Set fixed random number seed
torch.manual_seed(42)

# Resize and normalize the images
data_transforms = transforms.Compose([
                transforms.Resize(375),
                transforms.CenterCrop(375),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

# Create dataset
data_dir = '../folder_dataset'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms)
                for x in ['train', 'val', 'test']}

dataset = ConcatDataset([image_datasets['train'], image_datasets['val']])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                            shuffle=True, num_workers=4)
            for x in ['train', 'val', 'test']}
dataset_test_part = ConcatDataset([image_datasets['test']])

class_names = image_datasets['train'].classes

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

# Start print
print('--------------------------------')

# K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=100, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=100, sampler=test_subsampler)
    
    # Init the neural network
    network = CNN(len(class_names))
    network.to(device)
    # network.apply(reset_weights)
    
    # Initialize optimizer
    # optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.7)

    network.train()
    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):

        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
        
            # Get inputs
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = network(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 500))
                current_loss = 0.0
            
    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')
    
    # Saving the model
    save_path = f'./model-fold-{fold}.pth'
    torch.save(network.state_dict(), save_path)

    # Evaluationfor this fold
    network.eval()
    correct, total = 0, 0
    with torch.no_grad():

        # Iterate over the test data and generate predictions
        for i, data in enumerate(testloader, 0):

            # Get inputs
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Generate outputs
            outputs = network(inputs)

            # Set total and correct
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)
            
        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
        print(f'Average: {sum/len(results.items())} %')

# Test the model
model_test_acc = test_classifier(network, dataloaders['test'], device)
print("Final accuracy on test set: ", model_test_acc)