# AntID

This is the main package for performing individual recognition on harvestor ants.

### Prerequisites
Before being able to use antID, you must have a Flir, Spinnaker compatible, camera (can use other cameras
but some additional setup will be required) and have the `black_fly_camera` package and its
dependencies installed.

### Functionality
* `collect_data` is an executable for performing data collection of the ants. 
    * Upon running the script, you will be prompted to input the number of the ant ID you are 
    labelling. Once you are ready to move on to another ant, click the space bar and you will be prompted
    to end the session by clicking `q`, created a new video by clicking `n`, or to continue the 
    current video by clicking `c`.
    * When new videos are made they will be saved in the `data/videos` directory. The videos will be
    named with the following format: `antID_<id_number>.avi`. The ant id number will be incremented
    for every new video.