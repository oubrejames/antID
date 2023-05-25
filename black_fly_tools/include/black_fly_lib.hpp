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
        Spinnaker::ImageProcessor processor;
        Spinnaker::GenApi::CFloatPtr ptrExposuretime;
        Spinnaker::GenApi::CFloatPtr ptrFPS;

    public:
        BlackFlyCamera();
        ~BlackFlyCamera();

        
        Spinnaker::ImagePtr pResultImage;

        cv::Mat get_frame();

        void initialize_camera();

};
}
#endif
