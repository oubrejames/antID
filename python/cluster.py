from sklearn.cluster import HDBSCAN, DBSCAN
from datasets import AntsDataset
import torch
import torch.nn as nn
from torchvision import transforms
from networks import TripletNet, EmbeddingNet, FaceNet, EN2, EN4
import matplotlib.pyplot as plt
import numpy as np

"""
Using the embedding netwrok, this script clusters all ants in the testing set based on their 
predicted identities. It then tests to see how accuratly ants were predicted in the correct cluster.

TODO Work in progress
"""

######### PARAMETERS #########
embedding_network = EN4()
batch_size = 100
model_number = 46
gpu_id = "cuda:0"
threshold = 10
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


# Use both GPUs if declared
if gpu_parallel:
    model = nn.DataParallel(model, device_ids=[0,1]) # Use both GPUs
model.to(device)


# Load trained model and put into evaluation mode
trained_model_path = '../models/triplet_net_' + str(model_number) + '/best_model.pt'
model.load_state_dict(torch.load(trained_model_path, map_location=device))
model.eval()


# Declare array to hold embeddings
all_embeddings = []
original_labels = []


# Loop through all images in the dataset and get the embeddings
for images, labels in unseen_test_loader:
    images = images.to(device)
    batch_of_embeddings = model.get_embedding(images)

    # Save embeddings to an array of size (n_samples, n_features)
    for i, embedding in enumerate(batch_of_embeddings):
        # image = images[i].unsqueeze(0)
        label = labels[i]
        all_embeddings.append(embedding.detach().cpu().numpy())
        original_labels.append(label)

print("Created embedding array...")


# TODO combine this above and take out original labels
og_ant_ids = []
for label in original_labels:
    # Extract label number
    ant_id = label.split("_")[-1]
    og_ant_ids.append(int(ant_id))

clustered_class_embeddings = []
clustered_classes = []
# Loop through each embedding
for i in range(len(all_embeddings)):
    current_embedding = all_embeddings[i]
    cluster_matches_idxs = []
    match_distances = []
    match_count = 0

    # Compare emb against every cluster embedding
    for j in range(len(clustered_class_embeddings)):

        cluster_emb_to_test = clustered_class_embeddings[j]
        # dist = np.linalg.norm(current_embedding - cluster_emb_to_test, ord=2)
        dist = torch.linalg.vector_norm(torch.tensor(current_embedding - cluster_emb_to_test))**2
        # print(dist)
        if dist < threshold:
            match_count += 1
            cluster_matches_idxs.append(j)
            match_distances.append(dist)

    # If no match make a cluster
    if match_count == 0:
        clustered_class_embeddings.append(all_embeddings[i])
        clustered_classes.append(len(clustered_class_embeddings)-1)
        continue
    
    # Loop through all positive matches and choose best one
    smallest_dist = 9999999
    for p in range(len(cluster_matches_idxs)):
        if match_distances[p] < smallest_dist:
            smallest_dist = match_distances[p]
            best_match_idx = cluster_matches_idxs[p]
    
    # Average cluster embedding together
    clustered_class_embeddings[best_match_idx] = (clustered_class_embeddings[best_match_idx] + current_embedding)/2
    
    # Add class to list
    clustered_classes.append(best_match_idx)

# heat_map = np.zeros((len(clustered_classes), len(og_ant_ids), 1))
# for i in range(len(clustered_classes)):
#     for j in range(len(og_ant_ids)):
print("Number of cluster classes: ", len(clustered_class_embeddings))
distribution = []
for i in range(len(clustered_class_embeddings)):
    class_id = i + 1
    actual_ids_for_predicted_class = []
    for elm_idx, elm in enumerate(clustered_classes):
        if elm != class_id: # Purpose of this is to loop through only 1 of the predicted class labels at a time
            continue
        corresponding_ant_id = og_ant_ids[elm_idx]
        actual_ids_for_predicted_class.append(corresponding_ant_id)
    distribution.append([class_id, actual_ids_for_predicted_class])



print(clustered_classes)
for i in range(len(clustered_classes)):
    print(clustered_classes[i]," : ", og_ant_ids[i])

plt.scatter(og_ant_ids, clustered_classes)
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

# kmeans = KMeans(n_clustered_class_embeddings=10, random_state=0, n_init="auto").fit(embeddings)
# kmeans.labels_

# # Plot histogram of predicted labels
# # plt.hist(kmeans.labels_)
# # plt.savefig('kmeans.png')

# og_ant_ids = []
# for label in original_labels:
#     # Extract label number
#     ant_id = label.split("_")[-1]
#     og_ant_ids.append(int(ant_id))

# # bins=np.arange(min(og_ant_ids), max(og_ant_ids) + 2, 1)
# # plt.hist(og_ant_ids, bins = bins)
# # plt.savefig('labels_hist.png')


# clustered_classes_and_labels = []
# # Loop through all embeddings and save a list that is [predicted class, embedding]
# for i in range(len(embeddings)):
#     clustered_classes_and_labels.append([kmeans.labels_[i], og_ant_ids[i]])

# plt.scatter(og_ant_ids, kmeans.labels_)
# plt.savefig('scatter.png')\

# for i in range(len(clustered_classes_and_labels)):
#     print(clustered_classes_and_labels[i])