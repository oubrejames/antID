import torch
import matplotlib.pyplot as plt

def test_model(model, test_loader, device, threshold):
    """
    Loops through all images in the test_loader and calculates the true positive rate, true negative
    rate, false positive rate, false negative rate, and accuracy.
    

    Args:
        model (nn.Module): ant face recognition model
        test_loader (torch.utils.data.DataLoader): train dataloader
        device (torch.device): GPU or CPU
        threshold (float): threshold to determine if two images are the same

    Returns:
        None
    """

    # Load model to device
    model = model.to(device)

    # Initialize counts for metrics
    true_positives_count = 0
    true_negatives_count = 0
    false_positives_count = 0
    false_negatives_count = 0
    total_img_count = 0
    tested_count = 0

    # Variables to calculate average distance
    total_neg_dist= 0
    total_pos_dist = 0
    num_imgs_per_epoch = 0

    # Loop through all images in test_loader
    for anchors, positives, negatives, labels in test_loader:
        anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
        anchors_output, positives_output, negatives_output = model(anchors, positives, negatives)

        # Loop through all individual embeddings
        for i, anchor in enumerate(anchors_output):
            # Calculate the squared L2 distance between each output
            anchors_positives_dist = torch.linalg.vector_norm(anchors_output[i] - positives_output[i])**2
            anchors_negatives_dist = torch.linalg.vector_norm(anchors_output[i] - negatives_output[i])**2

            # Update total distances for average
            total_neg_dist += float(anchors_negatives_dist)
            total_pos_dist += float(anchors_positives_dist)

            # Predict if positives and anchors are the same
            if anchors_positives_dist < threshold:
                # Anchors and positve are predicted the same
                true_positives_count += 1
            else:
                # Anchors and positive are predicted different
                false_negatives_count += 1

            if anchors_negatives_dist > threshold:
                # Anchors and negative are predicted different
                true_negatives_count += 1
            else:
                # Anchors and negative are predicted the same
                false_positives_count += 1

            total_img_count += 1
            num_imgs_per_epoch += 1
        
        print("Avg pos dist for epoch: ", total_pos_dist/num_imgs_per_epoch)
        print("Avg neg dist for epoch: ", total_neg_dist/num_imgs_per_epoch)

        # Reset counts for next epoch
        total_neg_dist= 0
        total_pos_dist = 0
        num_imgs_per_epoch = 0

        # Caclulate metrics for each epoch tested
        if true_positives_count+false_positives_count:
            tp_rate = 100*true_positives_count/(true_positives_count+false_positives_count)
            fp_rate = 100*false_positives_count/(true_positives_count+false_positives_count)
        else:
            tp_rate = 0
            fp_rate = 0

        if true_negatives_count+false_negatives_count:
            tn_rate = 100*true_negatives_count/(true_negatives_count+false_negatives_count)
            fn_rate = 100*false_negatives_count/(true_negatives_count+false_negatives_count)
        else:
            tn_rate = 0
            fn_rate = 0

        accuracy = 100*(true_negatives_count+true_positives_count)/(true_negatives_count+\
                    true_positives_count+false_negatives_count+false_positives_count)

        print("TP Rate after ", tested_count, "epochs: ", tp_rate)
        print("TN Rate after ", tested_count, "epochs: ", tn_rate)
        print("FP Rate after ", tested_count, "epochs: ", fp_rate)
        print("FN Rate after ", tested_count, "epochs: ", fn_rate)
        print("Accuracy after", tested_count, "epochs: ", accuracy)
        print("------------------------------------------------------------------- \n")
        tested_count += 1

    # Print final metrics
    print("TP Rate Final: ", tp_rate)
    print("TN Rate Final: ", tn_rate)
    print("FP Rate Final: ", fp_rate)
    print("FN Rate Final: ", fn_rate)
    print("Final Accuracy: ", accuracy)
    print("Total number of testing images: ", total_img_count)

    return None

def test_thresholds(model, test_loader, device, path_to_folder):
    """Loops through all images in the test_loader and calculates the true positive rate, true negative rate,
    at different thresholds. Saves the results to a plot.

    Args:
        model (nn.Module): ant face recognition model
        test_loader (torch.utils.data.DataLoader): train dataloader
        device (torch.device): GPU or CPU
        path_to_folder (string): path to the folder to save the plot

    Returns:
        Average threshold (float): average threshold for the model
    """

    # Load  model to device
    model = model.to(device)

    # Initialize counts for metrics
    true_positives_count = 0
    true_negatives_count = 0
    false_positives_count = 0
    false_negatives_count = 0
    total_img_count = 1
    threshold_increment = 0

    # Flag to indicate if first epoch
    first_epoch = True

    # Create lists to save metrics and plot
    thresh_vals = []
    tp_vals = []
    tn_vals = []
    acc_vals = []

    # Variable to save best accuracy
    best_accuracy = 0

    # Variables to calculate average distance
    total_neg_dist = 0
    total_pos_dist = 0

    # Loop through all images in test_loader
    for anchors, positives, negatives, labels in test_loader:
        anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
        anchors_output, positives_output, negatives_output = model(anchors, positives, negatives)

        # Loop through all individual embeddings
        for i, anchor in enumerate(anchors_output):
            # Calculate the squared L2 distance between each output
            anchors_positives_dist = torch.norm(anchors_output[i] - positives_output[i])**2
            anchors_negatives_dist = torch.norm(anchors_output[i] - negatives_output[i])**2

            # Update total distances for average
            total_neg_dist += float(anchors_negatives_dist)
            total_pos_dist += float(anchors_positives_dist)

            # On first epoch, set threshold to a little less than average positive distance
            # Set threshold increment to be 5% of average positive distance
            if first_epoch:
                threshold = 0.8*float(anchors_positives_dist)
                threshold_increment = 0.05*float(anchors_positives_dist)

            # Predict if positives and anchors are the same
            if anchors_positives_dist < threshold:
                # Anchors and positve are predicted the same
                true_positives_count += 1
            else:
                # Anchors and positive are predicted different
                false_negatives_count += 1

            if anchors_negatives_dist > threshold:
                # Anchors and negative are predicted different
                true_negatives_count += 1
            else:
                # Anchors and negative are predicted the same
                false_positives_count += 1

            total_img_count += 1

        average_pos_dist = total_pos_dist/(total_img_count)
        average_neg_dist = total_neg_dist/(total_img_count)
        print("Average positives dist: ", average_pos_dist)
        print("Average negatives dist: ", average_neg_dist)

        print("Number of images tested: ", total_img_count)
        print("FN COUNT: ", false_negatives_count)
        print("FP COUNT: ", false_positives_count)
        print("TN COUNT: ", true_negatives_count)
        print("TP COUNT: ", true_positives_count)

        # Caclulate metrics for each epoch tested
        if true_positives_count+false_positives_count:
            tp_rate = 100*true_positives_count/(true_positives_count+false_positives_count)
            fp_rate = 100*false_positives_count/(true_positives_count+false_positives_count)
        else:
            tp_rate = 0
            fp_rate = 0

        if true_negatives_count+false_negatives_count:
            tn_rate = 100*true_negatives_count/(true_negatives_count+false_negatives_count)
            fn_rate = 100*false_negatives_count/(true_negatives_count+false_negatives_count)
        else:
            tn_rate = 0
            fn_rate = 0

        accuracy = 100*(true_negatives_count+true_positives_count)/(true_negatives_count+\
                    true_positives_count+false_negatives_count+false_positives_count)

        # Save threshold, tp_rate, tn_rate, and accuracy for plotting
        thresh_vals.append(threshold)
        tp_vals.append(tp_rate)
        tn_vals.append(tn_rate)
        acc_vals.append(accuracy)

        # Reset counts for next epoch
        true_positives_count = 0
        true_negatives_count = 0
        false_negatives_count = 0
        false_positives_count = 0

        # Increase threshold
        threshold += threshold_increment

        print("TP Rate at threshold = ", threshold, ": ", tp_rate)
        print("TN Rate at threshold = ", threshold, ": ", tn_rate)
        print("FP Rate at threshold = ", threshold, ": ", fp_rate)
        print("FN Rate at threshold = ", threshold, ": ", fn_rate)
        print("Accuracy at threshold = ", threshold, ": ", accuracy)

        print("------------------------------------------------------------------- \n")

    # Update best threshold
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

    average_pos_dist = total_pos_dist/(total_img_count+1)
    average_neg_dist = total_neg_dist/(total_img_count+1)
    print("Average positives dist: ", average_pos_dist)
    print("Average negatives dist: ", average_neg_dist)

    # average_thresh = average_pos_dist + (average_neg_dist - average_pos_dist)/2 # Keeping this commented out for now, stil deciding on best way to calculate output threshold
    plt.plot(thresh_vals, tp_vals, color = 'r', label = "tp_rate")
    plt.plot(thresh_vals, tn_vals, color = 'g', label = "tn_rate")
    plt.plot(thresh_vals, acc_vals, color = 'b', label = "accuracy")

    plt.xlabel('threshold')
    plt.ylabel('Rate')
    plt.legend()
    plt.savefig(path_to_folder + '/thresh_test.png')

    return best_threshold
