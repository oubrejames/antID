# AntID
This is the main repository for AntID. A biometric recognition system to classify indivdual harvestor ants
within a population. The purpose of this work to to create a system to help myrmecologist and biologists
by automating the process of identifying ants within a colony. Currently, the legnthy and difficult 
process of painting ants with different patterns is the norm (https://www.youtube.com/watch?v=uAQ5IKVpysc).

This repository currently consists of 2 packages, `black_fly_tools` and `antID`. 

`black_fly_tools` offers code that can be used to easily interface with the Flir Blackfly camera 
through the Spinnaker SDK.

`antID ` is the main package for performing individual recognition on the ants. 

### Experiment Setup
Below is a picture of the experiment setup. There are 4 main components to the setup. There are two
living area for the ants - a main living area and a separate temporary living area. The main living 
area is where the ants live typically and the temporary living area is where the ants are placed
after being maged. The temporary living area is connected to the main living area through the data 
collection area. The data collection area is where the camera is placed and where the ants are to walk
beneath the camera to be recorded. The last component are the ant control gates. These gates are used
to control the ants movement between the main living area, data collection area, and the temporary 
living area.

During data collection, one ant is allowed into the data collection area at a time. The ant is then
recorded for a while before being let into the temporary living area. A new video is then started using
the `data_collection` script in `antID `. This process is repeated until all ants have been recorded
and labelled.

<div align="center"><img src="https://github.com/oubrejames/antID/assets/46512429/eccbc9bd-b9c9-41ef-b49e-66b22ec4af2d" alt="antworld" width="500"/></div>

# Usage Instructions
_Note: You may have to adjust parameters on separate python scripts to match to ensure proper functionality_

### Data collection
* The `collect_data.cpp` file contains the script used to collect data. Allow one ant at a time in the
data collection area while the script is running. Once the ant has been filmed for a sufficient amount
of time click the space bar to pause the recording. You may then allow the ant to move into alternative 
living space to be separated from the unlabelled ants. Next, allow a new ant into the data collection
area and click `n` in terminal to start a new video. Videos will be saved in a folder, `labelled_vids`,
located in the root directory of the repository. This script can be found in the `antid` folder

### Creating the dataset
* With labelled videos collected you may then run `vid_to_imgs.py` to perform the YOLO object detection
and isolate frames where an ant is present. In the script you can alter parameters to choose where you want
to save the images and if you want to detect ant bodies or heads.

* Next, `format_dataset.py` or `folder_format_data.py` can be used to format the labelled images into
Pytorch usable datasets. The latter is used for the end to end CNN classifier.

### Training
* Tools for training a model can be found in the `trainer.py` files and the different networks I tried
out are present in the `networks.py` file.

* To train a triplet network or end to end classifier you can run the `TripletNetwork.py` or `classifier.py`
files. The tunable parameters are present at the top of each file.

### Testing
* Testing tools are located in the `tester.py` file and to test a model you can run the `test_model.py`
script. You may have to adjust the parameters at the top of the script. 


# Methodology

<p align = "center"><img src="https://github.com/oubrejames/antID/assets/46512429/c10ccb0f-a427-44be-b271-7a9c534163ca" /></p>
<p align = "center">System Pipeline</p>
 

The pipeline was structured as shown above. Videos would be processed using a YOLOV8 model trained to
detect ants find video frames where an ant was present. With ant images aquired, two different images 
would be put into the same network, known as Siamese network, and two embeddings would be output. I then 
obtain the distance between embeddings by calculating norm of the difference and if that norm is 
above a certain threshold the two ants in the images are different and if it is below, they are the same.


The model itself is a convolutional neural network and it outputs a 128 element feature vector. To train 
it I used a loss function known as triplet loss. Training with triplet loss can be understood intuitively 
by giving the trainer 3 images at each training step, an anchor, a positive, and a negative. The 
anchor is an image of any ant, the positive is then a different picture of the same ant, and the negative
is a picture of a different ant. During training, the model weights will be updated so that the
distance between anchor and positive embeddings is smaller than the distance between anchor and 
negative embeddngs.


Mathematically this works because the distances between the anchor positive pair minus the anchor 
negative pair plus some margin must be less than zero. Because the difference between distances must 
be negative, the anchor negative distance must be larger than the anchor positive distance. The 
margin is then used to try and push the distances from the anchor to positive and negative embeddings 
further from each other.

<p align = "center"><img src="https://github.com/oubrejames/antID/assets/46512429/9692bb97-9344-4ca4-9df5-629caa8e4fd8" /></p>
<p align = "center">Triplet Loss Function</p>

<p align = "center"><img src="https://github.com/oubrejames/antID/assets/46512429/5f83b33c-1a04-4203-a70d-9ffd9ba439ca" /></p>
<p align = "center">Cost Function with Triplet Loss</p>

# Results
The model was tested on two different test sets. One contained 2,357 different images of 45 ants that the model had seen in training and the other had 3,687 images of 10 unseen ants. On the dataset of previously
seen ants, the model achieved a true positive rate of 95.21%, a true negative rate of 99.6% and an accuracy
of 97.3%. On the dataset of unseen ants, the model achieved a true positive rate of 80.67%, a true negative rate of 93.68% and an accuracy of 86.04%. Accuracy here is defined as the number of true positives plus the number of true negative over the total number of predictions. 