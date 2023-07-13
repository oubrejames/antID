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
import csv

def test_model(model, test_loader, device):
    model = model.to(device)
    true_positive_count = 0
    true_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0
    total_count = 0
    tested_count = 0

    while tested_count < len(test_loader):
        for anchor, positive, negative, label in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_output, positive_output, negative_output = model(anchor, positive, negative)

            # Calculate the squared L2 distance between each output
            anchor_positive_dist = torch.norm(anchor_output - positive_output)**2
            anchor_negative_dist = torch.norm(anchor_output - negative_output)**2
            # print("Pos dist: ", anchor_positive_dist)
            # print("Neg dist: ", anchor_negative_dist)
            # Predict if positive and anchor are the same
            if anchor_positive_dist < 500:
                # Anchor and positve are predicted the same
                true_positive_count += 1
            else:
                false_negative_count += 1

            if anchor_negative_dist > 500:
                true_negative_count += 1
            else:
                false_positive_count += 1

            total_count += 1
            # True positive rate
        tp_rate = 100*true_positive_count/total_count
        tn_rate = 100*true_negative_count/total_count
        fp_rate = 100*false_positive_count/total_count
        fn_rate = 100*false_negative_count/total_count
        print("TP Rate after ", tested_count, "epochs: ", tp_rate)
        print("TN Rate after ", tested_count, "epochs: ", tn_rate)
        print("FP Rate after ", tested_count, "epochs: ", fp_rate)
        print("FN Rate after ", tested_count, "epochs: ", fn_rate)
        print("------------------------------------------------------------------- \n")
        tested_count += 1

    # True positive rate
    tp_rate = 100*true_positive_count/total_count
    tn_rate = 100*true_negative_count/total_count
    fp_rate = 100*false_positive_count/total_count
    fn_rate = 100*false_negative_count/total_count
    print("TP Rate Final: ", tp_rate)
    print("TN Rate Final: ", tn_rate)
    print("FP Rate Final: ", fp_rate)
    print("FN Rate Final: ", fn_rate)
    print("Total number of testing images: ", total_count)
    

    return None

# def main():
#     # Load model
#     model = torch.load('../bestloss0.pt')

    
# if __name__ == '__main__':
#     main()