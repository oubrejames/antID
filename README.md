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

![Antworld](https://github.com/oubrejames/antID/assets/46512429/0387fd0a-4e17-4ccb-9e59-8c364c2ea56e)

#### This repo is a work in progress.