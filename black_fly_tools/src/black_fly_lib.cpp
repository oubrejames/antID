#include "Spinnaker.h"
#include <black_fly_lib.hpp>

namespace bfc{
// Create a BlackFlyCamera constructer 
BlackFlyCamera::BlackFlyCamera(){
    try{
        // Create system instance
        system = Spinnaker::System::GetInstance();

        // Retrieve list of cameras from the system
        camList = system->GetCameras();

        // Ensure there are cameras plugged in
        if (camList.GetSize() == 0)
        {
            std::cout << "No cameras found." << std::endl;
        }

        // Get camera instance
        pCam = camList.GetByIndex(0);
        pCam->Init();

        // Retrieve camera FPS and display
        ptrFPS = pCam->GetNodeMap().GetNode("FrameRate");
        std::cout << "HEY" << std::endl;
        // std::cout << "Camera FPS: " << ptrFPS->GetValue() << std::endl;

        // Retrieve and display exposure time
        ptrExposuretime = pCam->GetNodeMap().GetNode("ExposureTime");
        float exposure_time = ptrExposuretime->GetValue();
        auto unit = ptrExposuretime->GetUnit();
        std::cout << "Exposure time " << exposure_time << " " << unit << std::endl;

        // Set processing to HQ linear (color)
        // processor.SetColorProcessing(Spinnaker::SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR);
    }
    catch (Spinnaker::Exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }
}

// Create a BlackFlyCamera destructor
BlackFlyCamera::~BlackFlyCamera(){
    if(pCam->IsStreaming())
        pCam->EndAcquisition();
    pResultImage->Release();
    camList.Clear();
    system->ReleaseInstance();
}

void BlackFlyCamera::initialize_camera(){
    // pCam->ExposureTimeSelector();
    // set_continuous_acquisition();
    pCam->BeginAcquisition();
}

cv::Mat BlackFlyCamera::get_frame(){
    Spinnaker::ImageProcessor processor;
    processor.SetColorProcessing(Spinnaker::SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR);

    pResultImage = pCam->GetNextImage(1000);

    Spinnaker::ImagePtr color_im = processor.Convert(pResultImage, Spinnaker::PixelFormat_BGR8);
    // cv::Mat frame = cv::Mat(pResultImage->GetHeight(), pResultImage->GetWidth(), (pResultImage->GetNumChannels() == 3) ? CV_8UC3 : CV_8UC1, pResultImage->GetData(), pResultImage->GetStride());
    cv::Mat frame = cv::Mat(color_im->GetHeight(), color_im->GetWidth(), (color_im->GetNumChannels() == 3) ? CV_8UC3 : CV_8UC1, color_im->GetData(), color_im->GetStride());

    pResultImage->Release();
    return frame;
}



}