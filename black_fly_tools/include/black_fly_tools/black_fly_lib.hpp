#ifndef BLACK_FLY_LAB_INCLUDE_GUARD_HPP
#define BLACK_FLY_LAB_INCLUDE_GUARD_HPP
/// \file
/// \brief Tools to use the BlackFly camera with Spinnaker SDK.

#include "Spinnaker.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace bfc
{

/// \brief A class to use the BlackFly camera with Spinnaker SDK.
class BlackFlyCamera
{
    private:
        /// @brief A Spinnaker system instance.
        Spinnaker::SystemPtr system;

        /// @brief List of Spinncaker cameras attached to the system.
        Spinnaker::CameraList camList;

        /// @brief A Spinnaker camera instance.
        Spinnaker::CameraPtr pCam;

        /// @brief A Spinnaker image processor instance to alter processing (i.e. mono8, BGR, etc).
        Spinnaker::ImageProcessor processor;

        /// @brief Pointer to Spinnaker exposure time variable.
        Spinnaker::GenApi::CFloatPtr ptrExposuretime;

        /// @brief Pointer to Spinnaker FPS variable.
        Spinnaker::GenApi::CFloatPtr ptrFPS;

    public:
        BlackFlyCamera();
        ~BlackFlyCamera();

        /// @brief A Spinnaker image instance.
        Spinnaker::ImagePtr pResultImage;

        /// @brief Obtain image frame from camera.
        /// @return Image frame as an OpenCV Mat.
        cv::Mat get_frame();

        /// @brief Start Spinncaker camera acquisition.
        void begin_acquisition();

        /// @brief Manually set camera exposure time. 
        /// @param exposure_time Exposure time in microseconds (Range 11 - 30,000,000).
        void set_exposure_time(float exposure_time);

        /// @brief Turn auto exposure on (continuous) or off.
        /// @param val Value of auto exposure (Continuous or off).
        void set_auto_exposure(const Spinnaker::GenICam::gcstring& val);
};
}
#endif
