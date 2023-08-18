This folder holds all the python code related to the AntID project including all the tools used to 
create datasets from videos of ants as well as create, train, and test different identification 
models. Links to the trained models, YOLO ant detection models, and datasets can be found in the main
repo README.

# Scripts
* `classifier.py` - Trains and tests a CNN classifier to predict ant identities on a closed dataset of ants
* MAKE WORK `cluster.py` - Uses the embedding network to cluster unseen ants into groups based off of their predicted identity 
* Delete passes in cleaned branch `datasets.py` - Holds custom Pyotrch dataset classes to use with ant images
* DELETE THIS FILE `dataset_stats.py
* `folder_format_data.py` - Loops through labelled images of ants and creates a dataset in Pytorch
ImageFolder format
* Delete this file `format_dataset_passes.py` - Loops through labelled ant images that are broken up
indivdual folders containing consecutive images of an ant doing one pass under the camera
* `format_dataset.py` - Loops through labelled images of ants and creates a dataset with each folder
containing images of one ant and a csv of all the image names and ant identities
* REMOVE `k_fold_classifier.py` - Performs k fold cross validation on the CNN classifier
* `model_sum.py` - Displays a summary of the model and its parameters
* CLEAN THIS FILE UP `networks.py` - Contains all network architectures used in the project
* `plot_loss.py` - Function to plot training loss
* `test_model.py` - Loops through validation set, finds highest accuracy threshold, tests model on
testing set of ants model is trained on, and tests model on set of ants the model has not been trained
on
* Delete passthru `tester.py` - Contains functions to test all models
* `trainer.py` - Contains functions needed for training different models
* REMOVE `Transfer_YOLO.py` - Script to change params and train a model transfer learned from the YOLO
ant detection model
* `TripletNetwork.py` - Script to train triplet networks to create embeddings with different parameters
* `vid_to_imgs.py` - Loops through labelled videos and saves frame with an ants present to a labelled
folder
* REMOVE `vid_to_single_pass.py` - Loops through labelled videos and saves frame with an ants present to a labelled
folder where individual passes of ants walking under the camera are grouped together in separate folders