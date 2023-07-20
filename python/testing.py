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
import matplotlib.pyplot as plt

def test_model(model, test_loader, device, threshold):
    model = model.to(device)
    true_positive_count = 0
    true_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0
    total_count = 0
    tested_count = 0

    total_neg_dist= 0
    total_pos_dist = 0

    while tested_count < len(test_loader):
        for anchor, positive, negative, label in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_output, positive_output, negative_output = model(anchor, positive, negative)

            # Calculate the squared L2 distance between each output
            anchor_positive_dist = torch.norm(anchor_output - positive_output)**2
            anchor_negative_dist = torch.norm(anchor_output - negative_output)**2

            total_neg_dist += float(anchor_negative_dist)
            total_pos_dist += float(anchor_positive_dist)

            # Predict if positive and anchor are the same
            if anchor_positive_dist < threshold:
                # Anchor and positve are predicted the same
                true_positive_count += 1
            else:
                false_negative_count += 1

            if anchor_negative_dist > threshold:
                true_negative_count += 1
            else:
                false_positive_count += 1

            total_count += 1
        
        print("Avg pos dist for epoch: ", total_pos_dist/100)
        print("Avg neg dist for epoch: ", total_neg_dist/100)
        total_neg_dist= 0
        total_pos_dist = 0
        # True positive rate
        tp_rate = 100*true_positive_count/(true_positive_count+false_positive_count)
        tn_rate = 100*true_negative_count/(true_negative_count+false_negative_count)
        fp_rate = 100*false_positive_count/(true_positive_count+false_positive_count)
        fn_rate = 100*false_negative_count/(true_negative_count+false_negative_count)
        accuracy = 100*(true_negative_count+true_positive_count)/(true_negative_count+\
            true_positive_count+false_negative_count+false_positive_count)
        print("TP Rate after ", tested_count, "epochs: ", tp_rate)
        print("TN Rate after ", tested_count, "epochs: ", tn_rate)
        print("FP Rate after ", tested_count, "epochs: ", fp_rate)
        print("FN Rate after ", tested_count, "epochs: ", fn_rate)
        print("Accuracy after", tested_count, "epochs: ", accuracy)
        print("------------------------------------------------------------------- \n")
        tested_count += 1

    # True positive rate
    tp_rate = 100*true_positive_count/(true_positive_count+false_positive_count)
    tn_rate = 100*true_negative_count/(true_negative_count+false_negative_count)
    fp_rate = 100*false_positive_count/(true_positive_count+false_positive_count)
    fn_rate = 100*false_negative_count/(true_negative_count+false_negative_count)
    accuracy = 100*(true_negative_count+true_positive_count)/(true_negative_count+\
        true_positive_count+false_negative_count+false_positive_count)
    print("TP Rate Final: ", tp_rate)
    print("TN Rate Final: ", tn_rate)
    print("FP Rate Final: ", fp_rate)
    print("FN Rate Final: ", fn_rate)
    print("Final Accuracy: ", accuracy)
    print("Total number of testing images: ", total_count)
    

    return None

def test_thresholds(model, test_loader, device, path_to_folder):
    model = model.to(device)
    true_positive_count = 0
    true_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0
    total_count = 0
    tested_count = 0
    epoch_count = 0
    threshold = 300
    thresh_vals = []
    tp_vals = []
    tn_vals = []
    best_threshold = threshold
    best_tp_rate = 0
    best_tn_rate = 0

    total_neg_dist = 0
    total_pos_dist = 0

    while tested_count < len(test_loader):
        for anchor, positive, negative, label in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_output, positive_output, negative_output = model(anchor, positive, negative)

            # Calculate the squared L2 distance between each output
            anchor_positive_dist = torch.norm(anchor_output - positive_output)**2
            anchor_negative_dist = torch.norm(anchor_output - negative_output)**2

            total_neg_dist += float(anchor_negative_dist)
            total_pos_dist += float(anchor_positive_dist)
            # print("Pos dist: ", anchor_positive_dist)
            # print("Neg dist: ", anchor_negative_dist)
            # Predict if positive and anchor are the same
            if anchor_positive_dist < threshold:
                # Anchor and positve are predicted the same
                true_positive_count += 1
            else:
                false_negative_count += 1

            if anchor_negative_dist > threshold:
                true_negative_count += 1
            else:
                false_positive_count += 1

            total_count += 1
            epoch_count += 1

        # Caclulate metrics for each epoch tested
        tp_rate = 100*true_positive_count/epoch_count
        tn_rate = 100*true_negative_count/epoch_count
        fp_rate = 100*false_positive_count/epoch_count
        fn_rate = 100*false_negative_count/epoch_count

        # Save threshold 
        thresh_vals.append(threshold)
        tp_vals.append(tp_rate)
        tn_vals.append(tn_rate)
        epoch_count = 0
        true_positive_count = 0
        true_negative_count = 0
        false_negative_count = 0
        false_positive_count = 0
        threshold += 5
        print("TP Rate at threshold = ", threshold, ": ", tp_rate)
        print("TN Rate at threshold = ", threshold, ": ", tn_rate)
        print("FP Rate at threshold = ", threshold, ": ", fp_rate)
        print("FN Rate at threshold = ", threshold, ": ", fn_rate)
        print("------------------------------------------------------------------- \n")
        tested_count += 1
        # Update best threshold
        if (tp_rate+tn_rate)/2 > (best_tp_rate+best_tn_rate)/2:
            best_threshold = threshold
            best_tn_rate = tn_rate
            best_tp_rate = tp_rate

    average_pos_dist = total_pos_dist/(total_count+1)
    average_neg_dist = total_neg_dist/(total_count+1)
    print("Average positive dist: ", average_pos_dist)
    print("Average negative dist: ", average_neg_dist)

    average_thresh = average_pos_dist + (average_neg_dist - average_pos_dist)/2
    plt.plot(thresh_vals, tp_vals, color = 'r', label = "tp_rate")
    plt.plot(thresh_vals, tn_vals, color = 'g', label = "tn_rate")
    plt.xlabel('threshold')
    plt.ylabel('Rate')
    plt.legend()
    plt.savefig(path_to_folder + '/thresh_test.png')

    return average_thresh
