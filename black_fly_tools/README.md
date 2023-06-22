# BlackFlyTools
---
Ths package contains a library, `black_fly_tools`, to interface with Flir Blackfly Camera via the
Spinnaker SDK. 
---

## Prerequisites
Package dependencies include:
* OpenCV4 (https://opencv.org/)
* Spinnaker SDK (https://www.flir.com/products/spinnaker-sdk/?vertical=machine+vision&segment=iis)
* Cmake (https://cmake.org/install/)
* In order for your computer to access the Blackfly you may need to configure the spinnaker udev
to give proper permissions
    * On my computer the udev file was located at `/etc/udev/rules.d/40-flir-spinnaker.rules`
    * Permissions were granted by adding the line `SUBSYSTEM=="usb", ATTRS{idVendor}=="1e10", MODE="0666"`


## Installation
Follow these steps to instaall BlackFly tools on your device:
1. Clone the repository. `git clone https://github.com/oubrejames/antID.git'
2. Go into the directory and add a build directory. `cd antID mkdir build`
3. Navigate to the build directory.
4. Install the package. `sudo make install`
5. The package can now be found in your `usr/local/` directory.

## Using BlackFlyTools
Using BlackFlyTools is fairly simple. Some key points are to:
1. Import the package using `#include <black_fly_tools/black_fly_lib.hpp>`.
2. Ensure camera is plugged in and create an instance of it, `bfc::BlackFlyCamera camera`.
3. Initialize a camera stream, `camera.begin_acquisition()`.
4. Get a frame from the stream as an OpenCV mat, `frame = camera.get_frame()`
5. Example code for recording and streaming can be found in the `src` directo45
