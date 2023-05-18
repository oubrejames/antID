#ifndef RIGID2D_INCLUDE_GUARD_HPP
#define RIGID2D_INCLUDE_GUARD_HPP
/// \file
/// \brief Two-dimensional rigid body transformations.

#include "Spinnaker.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace bfc
{

class BlackFlyCamera
{
    private:
        Spinnaker::SystemPtr system;
        Spinnaker::CameraList camList;
        Spinnaker::CameraPtr pCam;

    public:
        BlackFlyCamera();
        ~BlackFlyCamera();

        Spinnaker::ImagePtr pResultImage;

        int set_continuous_acquisition();
        cv::Mat get_frame();
        
        /*
        Gonna have to think about this more but I want this class to:
            1) Set up the camera object
            2) Make sure the camera is set to continuous acquisition
            3) Output frames in opencv format
        */
};
}
#endif
