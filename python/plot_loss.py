import matplotlib.pyplot as plt
import csv
  
val_loss = []
train_loss = []
epochs = []

with open('../loss.csv','r') as csvfile:
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
plt.savefig('loss_train.png')

plt.plot(epochs, val_loss, color = 'g',  label = "val_loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_val.png')


