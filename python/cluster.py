from sklearn.cluster import HDBSCAN
from datasets import AntsDataset
import torch
import torch.nn as nn
from torchvision import transforms
from networks import TripletNet, EmbeddingNet, FaceNet

######### PARAMETERS #########
embedding_network = EmbeddingNet()
batch_size = 100
model_number = 22
gpu_id = "cuda:1"
gpu_parallel = False
##############################

# Resize and normalize the images
data_transforms = transforms.Compose([
                transforms.Resize(375),
                transforms.CenterCrop(375),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

# Load dataset
unseen_dir = '../unseen_data'
unseen_csv = '../unseen_data/labels.csv'
unseen_dataset = AntsDataset(unseen_csv, unseen_dir, transform=data_transforms)
unseen_test_loader = torch.utils.data.DataLoader(unseen_dataset, batch_size=100, shuffle=True, num_workers=4)

# Load model
model = TripletNet(embedding_network)
device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")

if gpu_parallel:
    model = nn.DataParallel(model, device_ids=[0,1]) # Use both GPUs


# Load trained model and put into evaluation mode
trained_model_path = '../models/triplet_net_' + str(model_number) + '/best_model.pt'
model.load_state_dict(torch.load(trained_model_path, map_location=device))
model.eval()

# Declare array to hold embeddings
embeddings = []
labels = []

# Loop through all images in the dataset and get the embeddings
for images, labels in unseen_test_loader:
    
    # Save embeddings to an array of size (n_samples, n_features)
    for i in range(len(images)):
        image = images[i]
        label = labels[i]
        embeddings.append(model.get_embedding(image).detach().numpy())
        labels.append(label)

# Run HDBSCAN on the embeddings
hdb = HDBSCAN(min_cluster_size=10, min_samples=1)
hdb.fit(embeddings)
hdb_labels = hdb.labels_