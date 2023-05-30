#include "Spinnaker.h"
#include <black_fly_tools/black_fly_lib.hpp>

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

        // Initialize camera
        pCam->Init();

        // Retrieve camera FPS and display
        ptrFPS = pCam->GetNodeMap().GetNode("FrameRate");
        std::cout << "Camera FPS: " << ptrFPS->GetValue() << std::endl;

        // Retrieve and display exposure time
        ptrExposuretime = pCam->GetNodeMap().GetNode("ExposureTime");
        float exposure_time = ptrExposuretime->GetValue();
        auto unit = ptrExposuretime->GetUnit();
        std::cout << "Exposure time " << exposure_time << " " << unit << std::endl;
    }
    catch (Spinnaker::Exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }
}

// Create a BlackFlyCamera destructor
BlackFlyCamera::~BlackFlyCamera(){
    // End acquisition if camera is streaming
    if(pCam->IsStreaming())
        pCam->EndAcquisition();

    // Release image
    // pResultImage->Release();

    // Clear camera list before releasing system
    camList.Clear();

    // Release system instance
    system->ReleaseInstance();
}

void BlackFlyCamera::begin_acquisition(){
    // Begin acquiring images
    pCam->BeginAcquisition();
}

cv::Mat BlackFlyCamera::get_frame(){
    // Create image processor instance and set color processing
    // HQ Linear is noted in documentation to be well balanced for speed and resolution
    Spinnaker::ImageProcessor processor;
    processor.SetColorProcessing(Spinnaker::SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR);

    // Retrieve next image
    pResultImage = pCam->GetNextImage(1000);

    // Apply image processing
    Spinnaker::ImagePtr color_im = processor.Convert(pResultImage, Spinnaker::PixelFormat_BGR8);

    // Create OpenCV Mat from image
    cv::Mat frame = cv::Mat(color_im->GetHeight(), color_im->GetWidth(), (color_im->GetNumChannels() == 3) ? CV_8UC3 : CV_8UC1, color_im->GetData(), color_im->GetStride());

    // Release spinnaker image
    pResultImage->Release();
    return frame;
}

void BlackFlyCamera::set_auto_exposure(const Spinnaker::GenICam::gcstring& val){
    try{
        // Retrieve enumeration node from nodemap
        Spinnaker::GenApi::CEnumerationPtr ptrExposureAuto = pCam->GetNodeMap().GetNode("ExposureAuto");

        // Ensure auto exposure variable can be read and written
        if (Spinnaker::GenApi::IsReadable(ptrExposureAuto) && Spinnaker::GenApi::IsWritable(ptrExposureAuto))
        {
            // Set exposure auto to value
            Spinnaker::GenApi::CEnumEntryPtr ptrExposureAutoVal = ptrExposureAuto->GetEntryByName(val);
            if (Spinnaker::GenApi::IsReadable(ptrExposureAutoVal))
            {
                ptrExposureAuto->SetIntValue(ptrExposureAutoVal->GetValue());
            }
            else
            {
                std::cout << "Unable to set exposure auto to " << val << "..." << std::endl;
            }

            // Display auto exposure time setting
            std::cout << "Auto exposure set to " << ptrExposureAuto->GetCurrentEntry()->GetSymbolic() << std::endl;
            Spinnaker::GenApi::CFloatPtr ptrExposuretime = pCam->GetNodeMap().GetNode("ExposureTime");
            auto unit = ptrExposuretime->GetUnit();
            std::cout << "New exposure time " << ptrExposuretime->GetValue() << " " << unit << std::endl;
        }
        else
        {
            std::cout << "ExposureAuto not available..." << std::endl;
        }


    }
    catch (Spinnaker::Exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }
}

void BlackFlyCamera::set_exposure_time(float exposure_time){
    try{
        // Turn off auto exposure
        set_auto_exposure("Off");

        // Set the exposure time manually; exposure time recorded in microseconds
        // Retrieve and display exposure time
        Spinnaker::GenApi::CFloatPtr ptrExposuretime = pCam->GetNodeMap().GetNode("ExposureTime");
        ptrExposuretime->SetValue(exposure_time);
        auto unit = ptrExposuretime->GetUnit();
        std::cout << "New exposure time " << ptrExposuretime->GetValue() << " " << unit << std::endl;
    }
    catch (Spinnaker::Exception& e)
    {
        std::cout << "Error: " << e.what() << std::endl;
    }
}
}