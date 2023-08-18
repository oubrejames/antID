import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from datasets import TripletAntsDataset, TripletPasses
from networks import TripletNet, EmbeddingNet, FaceNet, EN2, CNN_EN, EN4, EN5, EN6, AntEmbeddingNet
from tester import test_model, test_thresholds, test_with_voting
import random
import numpy as np

######### PARAMETERS #########
embedding_network = AntEmbeddingNet()
batch_size = 70
model_number = 83
gpu_id = "cuda:0"
gpu_parallel = True
test_thresh = True
body_flag = True
if not test_thresh:
    best_threshold = 0.2
##############################

# Enable benchmarking for faster runtime
cudnn.benchmark = True


# Resize and normalize the images
data_transforms = transforms.Compose([
                # transforms.CenterCrop(375),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


# Choose correct dataset based on using body or not
if body_flag:
    # unseen_dir = '../unseen_body_data'
    # unseen_csv = '../unseen_body_data/labels.csv'
    unseen_dir = '../unseen_clean_data'
    unseen_csv = '../unseen_clean_data/labels.csv'
    seen_dir = '../ant_body_data'
    seen_csv = '../ant_body_data/labels.csv'
    model_folder = '../models/triplet_net_body_'
else:
    unseen_dir = '../unseen_data'
    unseen_csv = '../unseen_data/labels.csv'
    seen_dir = '../ant_face_data'
    seen_csv = '../ant_face_data/labels.csv'
    model_folder = '../models/triplet_net_'


# Create dataset and dataloader for unseen ant data
unseen_dataset = TripletAntsDataset(unseen_csv, unseen_dir, transform=data_transforms)
unseen_test_loader = torch.utils.data.DataLoader(unseen_dataset, batch_size=70, shuffle=True, num_workers=4)


# Create dataset and dataloader for seen ant data
seen_dataset = TripletAntsDataset(seen_csv, seen_dir, transform=data_transforms)


# Set random seed for reproducibility of split among different scripts
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


# Split dataset into train, validation, and test sets
train_size = int(0.8 * len(seen_dataset))
val_size = int(0.1 * len(seen_dataset))
test_size = len(seen_dataset) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(seen_dataset, [train_size, val_size, test_size])


# Create dataloaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
seen_val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
seen_test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)
dataloaders = {'train': train_loader, 'val': seen_val_loader, 'test': seen_test_loader, 'unseen': unseen_test_loader}


# Create model
model = TripletNet(embedding_network)
device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")

if gpu_parallel:
    model = nn.DataParallel(model, device_ids=[0,1]) # Use both GPUs


# Load trained model and put into evaluation mode
trained_model_path = model_folder + str(model_number) + '/best_model.pt'
model.load_state_dict(torch.load(trained_model_path, map_location=device))
model.eval()


# Create final model name for saving
final_model_folder = model_folder + str(model_number)


# Test model thresholds on validation data to find best threshold
if test_thresh:
    best_threshold, best_accuracy = test_thresholds(model, seen_val_loader, device, final_model_folder)
    print("Best threshold : ", best_threshold, "Accuracy: ", best_accuracy)


# Test model on seen test data
print("\n","*" * 100)
print("\nTesting model on seen test data...\n")
test_model(model, seen_test_loader, device, best_threshold)
print("-" * 50, '\n')


# Test model on unseen test data
print("\n","*" * 100)
print("\nTesting model on unseen test data...\n")
test_model(model, unseen_test_loader, device, best_threshold)
print("-" * 50, '\n')