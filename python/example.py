from networks import EN4
import pandas as pd
from PIL import Image
import os
import random
from networks import TripletNet, EN4
import torch
import torch.nn as nn
from torchvision import transforms
from vid_to_imgs import get_body_box
from ultralytics import YOLO
import csv
import cv2

######### PARAMETERS #########
embedding_network = EN4()
batch_size = 100
model_number = 78
gpu_id = "cuda:0"
threshold = 0.25
model_folder = '../models/triplet_net_body_'
##############################

class AntPredictor():
    """Class to use ease the use of the triplet network for predicting if two images are the same ant.
    """
    def __init__(self, model, weights_path, device, threshold, gpu_parallel = True, img_resize_dim = (512, 152)):
        """
        Args:
            model (nn.Module): Model to use for prediction (Tripletnet())
            weights_path (string): path to trained weights
            device (torch.device): GPU to work on
            threshold (float): threshold for deciding if two images are the same ant
            gpu_parallel (bool, optional): Flag for if the model was trained on 2 GPUs. Defaults to True.
            img_resize_dim (tuple, optional): Size to resize images to. Defaults to (512, 152).
        """
        self.model = model
        self.device = device

        if gpu_parallel:
            state_dict = torch.load(weights_path)
            new_state_dict = {}
            for key in state_dict:
                new_key = key.replace('module.','')
                new_state_dict[new_key] = state_dict[key]

            weights = new_state_dict
        else:
            weights = torch.load(weights_path, map_location=device)

        # Load model to device
        self.model = self.model.to(device)
        print("weights path", weights_path)

        self.model.load_state_dict(weights)
        self.model.eval()

        # Resize and normalize the images
        self.data_transforms = transforms.Compose([
                        transforms.Resize(img_resize_dim),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

        self.threshold = threshold


    def predict(self, image1, image2):
        """Take in two images and predict if they are the same ant.

        Args:
            image1 (np.array): image 1
            image2 (np.array): image 2

        Returns:
            True for same ant, False for different ant
        """
        # Convert to PIL image
        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)

        embd1 = get_embedding(image1)
        embd2 = get_embedding(image2)

        # Calculate the squared L2 distance between each output
        distance = torch.linalg.vector_norm(embd1 - embd2)**2

        return predict_embedding(embd1, embd2)

    def get_embedding(self, image1):
        """Return embedding for image

        Args:
            image1 (np.array): image to get embedding for

        Returns:
            tensor: embedding for image
        """
        image1 = Image.fromarray(image1)

        image1 = self.data_transforms(image1)

        # Add batch dimension
        image1 = torch.unsqueeze(image1, 0)

        image1 = image1.to(device)

        embd1 = self.model.get_embedding(image1)
        return embd1
    
    def predict_embedding(self, embd1, embd2):
        """Predict if two embeddings are the same ant

        Args:
            embd1 (tensor): embedding 1
            embd2 (tensor): embedding 2

        Returns:
            True for same ant, False for different ant
        """
        # Calculate the squared L2 distance between each output
        distance = torch.linalg.vector_norm(embd1 - embd2)**2

        # Predict if positives and anchors are the same
        if distance < self.threshold:
            return True
        else:
            return False

def calc_timestep(frame_number, fps):
    """Calulate the timestep for a given frame

    Args:
        frame_number (int): order of frame in video
        fps (int): frames per second of video

    Returns:
        float: timestep of frame
    """
    # s = f / fps
    return frame_number/fps


# Instantiate yolo model
yolo_model = YOLO("YOLO_Body/runs/detect/yolov8s_v8_25e3/weights/best.pt")

# Create predictor
model = TripletNet(embedding_network)
device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
trained_model_path = model_folder + str(model_number) + '/best_model.pt'
predictor = AntPredictor(model, trained_model_path, device, threshold)

# Load video
path_to_video = '../test_vid.mp4'

# Open video with OpenCV
cap= cv2.VideoCapture(path_to_video)
fps = cap.get(cv2.CAP_PROP_FPS)

# Predicted embeddings (index corresponds to ant id)
predicted_emeddings = []

# Counter for frame number
frame_count = 0

# Write headers for csv
row = ['ant_id', 'timestep']
with open("ant_timesteps.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow(row)

# Loop through all frames in video
while(1):
    ret, frame = cap.read()
    frame_count += 1
    if frame is None:
        break

    # Detect if ant is in frame
    bbox = get_body_box(frame, model = yolo_model)

    if bbox is None:
        continue

    # Crop image to head
    y1 = int(bbox[0][1])
    y2 = int(bbox[1][1])
    x1 = int(bbox[0][0])
    x2 = int(bbox[1][0])
    cropped_image = frame[y1:y2,x1:x2]

    frame_embd = predictor.get_embedding(cropped_image)

    # Loop through prediction embeddings
    for i, embd in enumerate(predicted_emeddings):
        if predictor.predict_embedding(frame_embd, embd):
            # Match found. Average embd
            predicted_emeddings[i] = (embd + frame_embd)/2

            # Get detected timestep
            ts = calc_timestep(frame_count, fps)
            row = [i, ts] # Write ant id and timestep to csv
            with open("ant_timesteps.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            break
    else:
        # Create new embedding
        predicted_emeddings.append(frame_embd)

        # Get detected timestep
        ts = calc_timestep(frame_count, fps)
        row = [len(predicted_emeddings), ts] # Write ant id and timestep to csv
        with open("ant_timesteps.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(row)

cap.release()
print("------------------------------------------------------------\n")