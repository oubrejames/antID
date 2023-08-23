from torchvision import models
from torchsummary import summary
from networks import TripletNet, EmbeddingNet, TransferYOLO, EN4
from ultralytics import YOLO
import torch
"""
Print out the model architecture for the a given model.
"""

print('Embedding net: \n ----------------------------')
model_emb = EmbeddingNet()
summary(model_emb, (3, 375, 375))

print('\n Printed Model \n --------------------------------')
print(model_emb)

print('\n Triplet net: \n ----------------------------')
model_trip = TripletNet(EmbeddingNet())
summary(model_trip,((3, 375, 375), (3, 375, 375), (3, 375, 375)))

print('\n Printed Model \n --------------------------------')
print(model_trip)

print('EN4: \n ----------------------------')
model_emb = EN4()
summary(model_emb, (3, 512, 152))

print('\n Printed Model \n --------------------------------')
print(model_emb)
