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
from datasets import TripletAntsDataset, AntsDataset
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
dataset = AntsDataset(csv_file, data_dir, transform=data_transforms['crop_norm'])

dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)


model = TripletNet(EmbeddingNet())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model = nn.DataParallel(model, device_ids=[0, 1]) # Use both GPUs

model.load_state_dict(torch.load('../models/triplet_net_9/best_model.pt'))
model.eval()
model.to(device)

# img_path = '../unseen_images/ant_17/im289.jpg'
# img = Image.open(img_path)
# img = dataset.transform(img)
# print("IMAGE SHAPE: ", img.shape)
# img = img.to(device)

# embedding = model.get_embedding(img)
# print("EMBD SHAPE: ". embedding.shape)

embd_label_arr = []

# Loop through every ant and save embedding and label to a list in form [[embd, lbl], ...]
for images, labels in dataloader:
    images = images.to(device)
    print("Image shape: ", images.shape)
    embedding = model(images, images, images)
    print("tensor shape:",embedding[0].shape)
    # print("tensor shape:",embedding[1].shape)
    # print("tensor shape:",embedding[2].shape)

    embedding = model.get_embedding(image).cpu().detach().numpy()
    print("embedding shape:",embedding.shape)
    embd_label_arr.append([embedding, label])

# embd_classes = []
# pred_embd_label_arr = []

# # Loop through all embeddings, compare current embedding from embd_label_arr to each embedding class
# # in embd_classes
# # If the current embedding is a positive match with a class embedding then add to list [[embd, label, class_pred],...]
# # and update the class embedding to average the two 
# new_class = True
# num_classes = 0
# for elm_pair in embd_label_arr:
#     embedding = elm_pair[0]
#     label = elm_pair[1]
    
#     # Loop through each embedding class
#     if embd_classes:
#         for class_embd_pair in embd_classes:
#             class_embedding = class_embd_pair[0]
#             embedding_class = class_embd_pair[1]
            
#             # Get the norm dist between the embedding and class embedding
#             dist_btw_embd = np.linalg.norm(embedding - class_embedding)**2

#             if dist_btw_embd < 500: # If there is a match TODO check if you get more than one match
#                 pred_embd_label_arr.append([embedding, label, embedding_class])
#                 print('POSITIVE')
#                 class_embd_pair[0] = (class_embd_pair[0]+embedding)/2
#                 new_class = False
#     else:
#         new_class = True
    
#     if new_class:
#         embd_classes.append([embedding, num_classes])
#         num_classes += 1
#         pred_embd_label_arr.append([embedding, label, num_classes])
