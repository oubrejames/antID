#include "Spinnaker.h"
#include <black_fly_lib.hpp>

namespace bfc{
// Create a BlackFlyCamera constructer 
BlackFlyCamera::BlackFlyCamera(){
    system = Spinnaker::System::GetInstance();
    camList = system->GetCameras(); 
    pCam = camList.GetByIndex(0);
    pResultImage = pCam->GetNextImage();
}

// Create a BlackFlyCamera destructor
BlackFlyCamera::~BlackFlyCamera(){
    pResultImage->Release();
    camList.Clear();
    system->ReleaseInstance();
}

int BlackFlyCamera::set_continuous_acquisition(){
        Spinnaker::GenApi::INodeMap & nodeMap = pCam->GetNodeMap();
        Spinnaker::GenApi::CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
        if (!Spinnaker::GenApi::IsAvailable(ptrAcquisitionMode) || !Spinnaker::GenApi::IsWritable(ptrAcquisitionMode))
        {
            std::cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << std::endl << std::endl;
            return -1;
        }

        // Retrieve entry node from enumeration node
        Spinnaker::GenApi::CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
        if (!Spinnaker::GenApi::IsAvailable(ptrAcquisitionModeContinuous) || !Spinnaker::GenApi::IsReadable(ptrAcquisitionModeContinuous))
        {
            std::cout << "Unable to set acquisition mode to continuous (entry retrieval). Aborting..." << std::endl << std::endl;
            return -1;
        }

        // Retrieve integer value from entry node
        const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

        // Set integer value from entry node as new value of enumeration node
        ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);
}

cv::Mat BlackFlyCamera::get_frame(){
    pResultImage = pCam->GetNextImage();
    cv::Mat frame = cv::Mat(pResultImage->GetHeight(), pResultImage->GetWidth(), (pResultImage->GetNumChannels() == 3) ? CV_8UC3 : CV_8UC1, pResultImage->GetData(), pResultImage->GetStride());
    return frame;
}
}