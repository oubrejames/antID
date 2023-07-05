import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings
from matplotlib import pyplot as plt
import torch.nn as nn
from tempfile import TemporaryDirectory
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
        # id = id.split("_")[-1] # Get the last character of the id (the number) 
        sample = {'id':id, 'image': image}

        if self.transform:
            sample = self.transform(sample)

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

        # h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size

        # new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (self.output_size, self.output_size))

        return {'image': img, 'id': id}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, id = sample['image'], sample['id']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'id': id}

# Creating a CNN class
class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

############################
############################
############################
############
# Define relevant variables for the ML task
batch_size = 64
num_classes = 11
learning_rate = 0.001
num_epochs = 20
############

face_dataset = AntFaceDataset(csv_file='../ant_faces_dataset/faces/labels.csv',
                                    root_dir='../ant_faces_dataset/faces/',
                                    transform=transforms.Compose([
                                        Rescale(375),
                                        ToTensor()
                                    ]))



train_size = int(0.8 * len(face_dataset))
val_size = int(0.1 * len(face_dataset))
test_size = len(face_dataset) - train_size - val_size

train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(face_dataset, [train_size, test_size, val_size])

train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

val_loader = DataLoader(val_dataset, 
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
datasizes = {'train': train_size, 'val': val_size, 'test': test_size}
class_names = face_dataset.classes

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

print("Class", classes[0])

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ConvNeuralNet(num_classes)

# Set Loss function with criterion
criterion = nn.CrossEntropyLoss()

# Set optimizer with optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

total_step = len(train_loader)


############ Pyotrch Tutorial ############  
# Create a temporary directory to save training checkpoints
with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

    torch.save(model.state_dict(), best_model_params_path)
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(torch.load(best_model_params_path))






# #######
# ####
# # https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/

# # Training loop
# for epoch in range(num_epochs):
# 	#Load in the data in batches using the train_loader object
#     for i, element in enumerate(train_loader):
#         # Move tensors to the configured device
#         images = element['image'].to(device)
#         labels = element['id'].to(device)
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# # Test
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in train_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
    
#     print('Accuracy of the network on the {} train images: {} %'.format(train_size, 100 * correct / total))
