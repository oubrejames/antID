from sklearn.cluster import HDBSCAN, DBSCAN
from datasets import AntsDataset
import torch
import torch.nn as nn
from torchvision import transforms
from networks import TripletNet, EmbeddingNet, FaceNet
import matplotlib.pyplot as plt
import numpy as np

######### PARAMETERS #########
embedding_network = EmbeddingNet()
batch_size = 100
model_number = 22
gpu_id = "cuda:1"
threshold = 2.5
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
model.to(device)

# Load trained model and put into evaluation mode
trained_model_path = '../models/triplet_net_' + str(model_number) + '/best_model.pt'
model.load_state_dict(torch.load(trained_model_path, map_location=device))
model.eval()

# Declare array to hold embeddings
embeddings = []
list_of_original_labels = []

# Loop through all images in the dataset and get the embeddings
for images, labels in unseen_test_loader:
    images = images.to(device)
    batch_of_embeddings = model.get_embedding(images)

    # Save embeddings to an array of size (n_samples, n_features)
    for i, embedding in enumerate(batch_of_embeddings):
        # image = images[i].unsqueeze(0)
        label = labels[i]
        embeddings.append(embedding.detach().cpu().numpy())
        list_of_original_labels.append(label)

print("Created embedding array...")


ant_ids = []
for label in list_of_original_labels:
    # Extract label number
    ant_id = label.split("_")[-1]
    ant_ids.append(int(ant_id))

clusters = []
classes = []
# Loop through each embedding
for i in range(len(embeddings)):
    emb = embeddings[i]
    positives = []
    pos_dists = []
    match_count = 0

    # Compare emb against every cluster embedding
    for j in range(len(clusters)):
        if i == j:
            continue

        test_emb = clusters[j]
        dist = np.linalg.norm(emb - test_emb, ord=2)

        if dist < threshold:
            match_count += 1
            positives.append(j)
            pos_dists.append(dist)
    
    # If no match make a cluster
    if match_count == 0:
        clusters.append(embeddings[i])
        classes.append(len(clusters))
        continue
    
    # Loop through all positive matches and choose best one
    smallest_dist = 9999999
    for p in range(len(positives)):
        if pos_dists[p] < smallest_dist:
            smallest_dist = pos_dists[p]
            best_match = positives[p] # positives[p] is the cluster index of best match
    
    # Average cluster embedding together
    clusters[best_match] = (clusters[best_match] + emb)/2
    
    # Add class to list
    classes.append(best_match)

plt.scatter(ant_ids, classes)
plt.savefig('scatter.png')



#####################################################################################3
# dbscan = DBSCAN(eps=0.7, min_samples=5, metric = 'l2').fit(embeddings)
# print(dbscan.labels_)

# # Run HDBSCAN on the embeddings
# hdb = HDBSCAN(min_cluster_size=10, min_samples=1, metric='manhattan')
# hdb.fit(embeddings)
# hdb_labels = hdb.labels_
# for lbl in hdb_labels:
#     print(lbl)

# from sklearn.cluster import KMeans

# kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(embeddings)
# kmeans.labels_

# # Plot histogram of predicted labels
# # plt.hist(kmeans.labels_)
# # plt.savefig('kmeans.png')

# ant_ids = []
# for label in list_of_original_labels:
#     # Extract label number
#     ant_id = label.split("_")[-1]
#     ant_ids.append(int(ant_id))

# # bins=np.arange(min(ant_ids), max(ant_ids) + 2, 1)
# # plt.hist(ant_ids, bins = bins)
# # plt.savefig('labels_hist.png')


# classes_and_labels = []
# # Loop through all embeddings and save a list that is [predicted class, embedding]
# for i in range(len(embeddings)):
#     classes_and_labels.append([kmeans.labels_[i], ant_ids[i]])

# plt.scatter(ant_ids, kmeans.labels_)
# plt.savefig('scatter.png')\

# for i in range(len(classes_and_labels)):
#     print(classes_and_labels[i])