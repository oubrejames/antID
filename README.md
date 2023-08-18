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

<img src="https://github.com/oubrejames/antID/assets/46512429/0387fd0a-4e17-4ccb-9e59-8c364c2ea56e" alt="antworld" width="500"/>

# Methodology
Just copy script from video and reword

# Usage Instructions
_ Note: You may have to adjust parameters on separate python scripts to match to ensure proper functionality_

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

# Results
