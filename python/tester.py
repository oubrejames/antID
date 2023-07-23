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
    true_positives_count = 0
    true_negatives_count = 0
    false_positives_count = 0
    false_negatives_count = 0
    total_img_count = 0
    tested_count = 0

    total_neg_dist= 0
    total_pos_dist = 0
    image_epoch_count = 0

    while tested_count < len(test_loader):
        for anchors, positives, negatives, labels in test_loader:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            anchors_output, positives_output, negatives_output = model(anchors, positives, negatives)
            # print("ANCHORS SHAPE: ", anchors.shape) === 100, 3, 375, 375
            # print("TEST LOADER LEN: ", len(test_loader)) === 28
            # Loop through all individual embeddings
            for i, anchor in enumerate(anchors_output):
                # Calculate the squared L2 distance between each output
                anchors_positives_dist = torch.linalg.vector_norm(anchors_output[i] - positives_output[i])**2
                anchors_negatives_dist = torch.linalg.vector_norm(anchors_output[i] - negatives_output[i])**2

                total_neg_dist += float(anchors_negatives_dist)
                total_pos_dist += float(anchors_positives_dist)

                # Predict if positives and anchors are the same
                if anchors_positives_dist < threshold:
                    # anchors and positve are predicted the same
                    true_positives_count += 1
                else:
                    false_negatives_count += 1

                if anchors_negatives_dist > threshold:
                    true_negatives_count += 1
                else:
                    false_positives_count += 1

                total_img_count += 1
                image_epoch_count += 1
        
        print("Avg pos dist for epoch: ", total_pos_dist/image_epoch_count)
        print("Avg neg dist for epoch: ", total_neg_dist/image_epoch_count)
        total_neg_dist= 0
        total_pos_dist = 0
        image_epoch_count = 0

        # True positives rate
        tp_rate = 100*true_positives_count/(true_positives_count+false_positives_count)
        tn_rate = 100*true_negatives_count/(true_negatives_count+false_negatives_count)
        fp_rate = 100*false_positives_count/(true_positives_count+false_positives_count)
        fn_rate = 100*false_negatives_count/(true_negatives_count+false_negatives_count)
        accuracy = 100*(true_negatives_count+true_positives_count)/(true_negatives_count+\
            true_positives_count+false_negatives_count+false_positives_count)
        print("TP Rate after ", tested_count, "epochs: ", tp_rate)
        print("TN Rate after ", tested_count, "epochs: ", tn_rate)
        print("FP Rate after ", tested_count, "epochs: ", fp_rate)
        print("FN Rate after ", tested_count, "epochs: ", fn_rate)
        print("Accuracy after", tested_count, "epochs: ", accuracy)
        print("------------------------------------------------------------------- \n")
        tested_count += 1

    # True positives rate
    tp_rate = 100*true_positives_count/(true_positives_count+false_positives_count)
    tn_rate = 100*true_negatives_count/(true_negatives_count+false_negatives_count)
    fp_rate = 100*false_positives_count/(true_positives_count+false_positives_count)
    fn_rate = 100*false_negatives_count/(true_negatives_count+false_negatives_count)
    accuracy = 100*(true_negatives_count+true_positives_count)/(true_negatives_count+\
        true_positives_count+false_negatives_count+false_positives_count)
    print("TP Rate Final: ", tp_rate)
    print("TN Rate Final: ", tn_rate)
    print("FP Rate Final: ", fp_rate)
    print("FN Rate Final: ", fn_rate)
    print("Final Accuracy: ", accuracy)
    print("Total number of testing images: ", total_img_count)
    

    return None

def test_thresholds(model, test_loader, device, path_to_folder):
    model = model.to(device)
    true_positives_count = 0
    true_negatives_count = 0
    false_positives_count = 0
    false_negatives_count = 0
    total_img_count = 0
    tested_count = 0
    epoch_count = 0
    threshold = 0.5
    thresh_vals = []
    tp_vals = []
    tn_vals = []
    acc_vals = []
    best_threshold = threshold
    best_tp_rate = 0
    best_tn_rate = 0

    total_neg_dist = 0
    total_pos_dist = 0

    while tested_count < len(test_loader):
        for anchors, positives, negatives, labels in test_loader:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            anchors_output, positives_output, negatives_output = model(anchors, positives, negatives)

            # Loop through all individual embeddings
            for i, anchor in enumerate(anchors_output):
                # print("Output size: ", anchor.shape)

                # Calculate the squared L2 distance between each output
                anchors_positives_dist = torch.norm(anchors_output[i] - positives_output[i])**2
                anchors_negatives_dist = torch.norm(anchors_output[i] - negatives_output[i])**2

                total_neg_dist += float(anchors_negatives_dist)
                total_pos_dist += float(anchors_positives_dist)

                # print("POS:" , anchors_positives_dist)
                # print("NEG:", anchors_negatives_dist)
                # Predict if positives and anchors are the same
                if anchors_positives_dist/(10**13) < threshold:
                    # anchors and positve are predicted the same
                    true_positives_count += 1
                else:
                    false_negatives_count += 1

                if anchors_negatives_dist/(10**13) > threshold:
                    true_negatives_count += 1
                else:
                    false_positives_count += 1

                total_img_count += 1
            average_pos_dist = total_pos_dist/(total_img_count)
            average_neg_dist = total_neg_dist/(total_img_count)
            print("Average positives dist: ", average_pos_dist)
            print("Average negatives dist: ", average_neg_dist)
            epoch_count += 1

            print("FN COUNT: ", false_negatives_count)
            print("FP COUNT: ", false_positives_count)
            print("TN COUNT: ", true_negatives_count)
            print("TP COUNT: ", true_positives_count)
            # Caclulate metrics for each epoch tested
            tp_rate = 100*true_positives_count/(true_positives_count+false_positives_count +0.0000001)
            tn_rate = 100*true_negatives_count/(true_negatives_count+false_negatives_count +0.0000001)
            fp_rate = 100*false_positives_count/(true_positives_count+false_positives_count +0.0000001)
            fn_rate = 100*false_negatives_count/(true_negatives_count+false_negatives_count +0.0000001)
            accuracy = 100*(true_negatives_count+true_positives_count)/(true_negatives_count+\
                        true_positives_count+false_negatives_count+false_positives_count + 0.0000001)
            # Save threshold 
            thresh_vals.append(threshold)
            tp_vals.append(tp_rate)
            tn_vals.append(tn_rate)
            acc_vals.append(accuracy)
            epoch_count = 0
            true_positives_count = 0
            true_negatives_count = 0
            false_negatives_count = 0
            false_positives_count = 0
            threshold += 0.25
            print("TP Rate at threshold = ", threshold, ": ", tp_rate)
            print("TN Rate at threshold = ", threshold, ": ", tn_rate)
            print("FP Rate at threshold = ", threshold, ": ", fp_rate)
            print("FN Rate at threshold = ", threshold, ": ", fn_rate)
            print("Accuracy at threshold = ", threshold, ": ", accuracy)

            print("------------------------------------------------------------------- \n")
        tested_count += 1
        # Update best threshold
        if (tp_rate+tn_rate)/2 > (best_tp_rate+best_tn_rate)/2:
            best_threshold = threshold
            best_tn_rate = tn_rate
            best_tp_rate = tp_rate

    average_pos_dist = total_pos_dist/(total_img_count+1)
    average_neg_dist = total_neg_dist/(total_img_count+1)
    print("Average positives dist: ", average_pos_dist)
    print("Average negatives dist: ", average_neg_dist)

    average_thresh = average_pos_dist + (average_neg_dist - average_pos_dist)/2
    plt.plot(thresh_vals, tp_vals, color = 'r', label = "tp_rate")
    plt.plot(thresh_vals, tn_vals, color = 'g', label = "tn_rate")
    plt.plot(thresh_vals, acc_vals, color = 'b', label = "accuracy")

    plt.xlabel('threshold')
    plt.ylabel('Rate')
    plt.legend()
    plt.savefig(path_to_folder + '/thresh_test.png')

    return average_thresh
