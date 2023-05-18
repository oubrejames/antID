#ifndef RIGID2D_INCLUDE_GUARD_HPP
#define RIGID2D_INCLUDE_GUARD_HPP
/// \file
/// \brief Two-dimensional rigid body transformations.

#include "Spinnaker.h"

class BlackFlyCamera
{
    private:
        Spinnaker::SystemPtr system;
        Spinnaker::CameraList camList;
        Spinnaker::CameraPtr pCam;
        
    public:
        Spinnaker::ImagePtr pResultImage;


}

#endif
