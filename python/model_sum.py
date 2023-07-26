from torchvision import models
from torchsummary import summary
from networks import TripletNet, EmbeddingNet, TransferYOLO
from ultralytics import YOLO
import torch
"""
Print out the model architecture for the a given model.
"""

# print('Embedding net: \n ----------------------------')
# model_e = EmbeddingNet()
# summary(model_e, (3, 375, 375))

# print('\n Triplet net: \n ----------------------------')

# model = TripletNet(EmbeddingNet())
# summary(model,((3, 375, 375), (3, 375, 375), (3, 375, 375)))

# print('\n Printed Model \n --------------------------------')
# print(model_e)

model = TransferYOLO()
print(model)

summary(model, (3,375,375))