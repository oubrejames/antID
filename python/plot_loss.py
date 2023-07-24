import matplotlib.pyplot as plt
import csv

def plot_loss(path_to_folder):
    """Loop through the loss.csv file and plot the loss for each epoch.

    Args:
        path_to_folder (string): path to the folder containing the loss.csv file
    """

    val_loss = []
    train_loss = []
    epochs = []
    path_to_loss = path_to_folder + "/loss.csv"

    with open(path_to_loss,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')

        count = -1
        for row in plots:
            if count > 0:
                val_loss.append((float(row[0])))
                train_loss.append((float(row[1])))
                epochs.append(count)
            count += 1

    plt.plot(epochs, train_loss, color = 'r', label = "train_loss")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path_to_folder + '/loss_train.png')

    plt.plot(epochs, val_loss, color = 'g',  label = "val_loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path_to_folder + '/loss_val.png')