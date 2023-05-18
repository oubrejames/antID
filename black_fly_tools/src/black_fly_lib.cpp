#include "Spinnaker.h"
#include "black_fly_lib.hpp"

// Create a BlackFlyCamera constructer 
BlackFlyCamera::BlackFlyCamera(): 
    system{Spinnaker::System::GetInstance();}, 
    camList{system->GetCameras();}, 
    pCam{camList.GetByIndex(0)},
    pResultImage{pCam->GetNextImage()}{}

// Create a BlackFlyCamera destructor
BlackFlyCamera::~BlackFlyCamera() {
    pResultImage->Release();
}