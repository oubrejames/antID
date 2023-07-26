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
from networks import TripletNet, EmbeddingNet, FaceNet, EN2, TransferYOLO, EN3
from trainer import fit_triplet
from tester import test_model
import shutil
from plot_loss import plot_loss
from trainer import EarlyStopper
import pandas as pd
import matplotlib.pyplot as plt
"""
This script trains a triplet network on the ant face dataset.
"""

######### PARAMETERS #########
embedding_network = EN3()
batch_size = 100
loss_margin = 0.5
loss_p = 2
learn_rate = 0.01
gpu_id = "cuda:1"
gpu_parallel = False
scheduler_step_size = 7
scheduler_gamma = 0.1
num_epochs = 200
early_stopper = EarlyStopper(patience=15, min_delta=0.001)
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



#################################################################################################
#################################################################################################
#################### TESTING WITH BALANCED DATA #############################
#################################################################################################
#################################################################################################

# #Upload the data and save it to a Dataframe.
# df = pd.read_csv( '../ant_face_data/labels.csv', header=None)
# df=pd.DataFrame(df)
# print("df size: ", df.shape)

# # Check the class distribution. (Hint: Use .value_counts attribute and 
# # remember that the labels are at the last column of the dataframe)
# results = df[0]


# # Ants 11, 12, 16, and 26 have much more instances than the other training data and ant 8 has barely
# # any.
# # Make the dataset more balanced by sampling ants 11, 12, 16, and 26 and removing ant 8
# missing_ants = df.drop(df[df[0] == 'ant_11'])
# # The value you want to remove from the DataFrame
# target_classes = ["ant_11", "ant_12", "ant_16", "ant_26", "ant_8"]

# filtered_df = df
# for target_class in target_classes:
#     # Boolean indexing to filter out rows with the target_class
#     filtered_df = filtered_df[filtered_df[0] != target_class]

# # print(missing_ants)
# # missing_ants.drop(missing_ants[missing_ants[0] == 'ant_12'])
# # missing_ants = missing_ants.drop(missing_ants[missing_ants[0] == 'ant_16'])
# # missing_ants = missing_ants.drop(missing_ants[missing_ants[0] == 'ant_26'])
# # missing_ants = missing_ants.drop(missing_ants[missing_ants[0] == 'ant_8'])

# all_11 = df[df[0] == 'ant_11']
# all_12 = df[df[0] == 'ant_12']
# all_16 = df[df[0] == 'ant_16']
# all_26 = df[df[0] == 'ant_26']

# smaller_11 = all_11.sample(860)
# smaller_12 = all_12.sample(860)
# smaller_16 = all_16.sample(860)
# smaller_26 = all_26.sample(860)

# # # balanced_df = balanced_df.drop(df[df[0] == 'ant_12'].sample(860).index)
# # # balanced_df = balanced_df.drop(df[df[0] == 'ant_16'].sample(860).index)
# # # balanced_df = balanced_df.drop(df[df[0] == 'ant_26'].sample(860).index)
# # # balanced_df = balanced_df.drop(df[df[0] == 'ant_8'])
# balanced_df = pd.concat([smaller_11, smaller_12, smaller_16, smaller_26, filtered_df], axis=0)
# csv_file = balanced_df
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################


# Create dataset
data_dir = '../ant_face_data'
csv_file = '../ant_face_data/labels.csv'
dataset = TripletAntsDataset(csv_file, data_dir, transform=data_transforms)


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
optimizer = optim.Adam(model.parameters(), lr=learn_rate)
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
final_model_folder = "../models/triplet_net_" + str(latest_model_seq)
final_model_path = final_model_folder + "/best_model.pt"


# Save model
os.makedirs("../models", exist_ok=True)
os.makedirs(final_model_folder)
torch.save(final_model.state_dict(), final_model_path)


# Copy the loss to folder
path_to_loss = "../loss.csv" # Loss gets overwritten each time, so this is fine
shutil.copy(path_to_loss, final_model_folder)
plot_loss(final_model_folder)