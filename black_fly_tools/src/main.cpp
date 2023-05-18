#include <opencv2/opencv.hpp>
#include <black_fly_lib.hpp>
#include "Spinnaker.h"

int main() {

    Spinnaker::SystemPtr system = Spinnaker::System::GetInstance();
    Spinnaker::CameraList camList = system->GetCameras();

    if (camList.GetSize() == 0)
    {
        std::cout << "No cameras found." << std::endl;
        return -1;
    }

    Spinnaker::CameraPtr pCam = camList.GetByIndex(0);

// Initialize the camera
    pCam->Init();
    pCam->BeginAcquisition();

    std::cout << "Get next image" << std::endl;
    Spinnaker::ImagePtr pResultImage = pCam->GetNextImage(1000);
    pResultImage->Release();
    std::cout << "Got next image" << std::endl;

    cv::Mat frame = cv::Mat(pResultImage->GetHeight(), pResultImage->GetWidth(), (pResultImage->GetNumChannels() == 3) ? CV_8UC3 : CV_8UC1, pResultImage->GetData(), pResultImage->GetStride());


  while (true) {
    pResultImage = pCam->GetNextImage(1000);
    std::cout << "YAGA" << std::endl;
    cv::Mat frame = cv::Mat(pResultImage->GetHeight(), pResultImage->GetWidth(), (pResultImage->GetNumChannels() == 3) ? CV_8UC3 : CV_8UC1, pResultImage->GetData(), pResultImage->GetStride());
    pResultImage->Release();
    if (frame.empty()) {
      std::cout << "Failed to capture a frame." << std::endl;
      break;
    }

    cv::imshow("Video", frame); // Display the frame

    // Exit the loop if the 'Esc' key is pressed
    if (cv::waitKey(1) == 27)
      break;
  }

  cv::destroyAllWindows(); // Close all windows

  return 0;
}

// int main() {

//     // Create a BlackFlyCamera object
//     std::cout << "Create camera object" << std::endl;
//     // auto system = Spinnaker::System::GetInstance();
//     // auto camList = system->GetCameras(); 
//     // auto pCam = camList.GetByIndex(0);

//     // std::cout << "Number of cameras detected: " << camList.GetSize() << std::endl << std::endl;

//     bfc::BlackFlyCamera cam;

//     // Set the camera to continuous acquisition mode
//     std::cout << "Camera object created" << std::endl;
//     // std::cout << "Set camera to continuous acquisition" << std::endl;
//     // cam.set_continuous_acquisition();
//     // std::cout << "Continuous acquisition set" << std::endl;



//   while (true) {
//     cv::Mat frame;
//     std::cout << "Capture frame from camera" << std::endl;

//     frame = cam.get_frame(); // Read a new frame from the camera
//     std::cout << "Got frame" << std::endl;

//     if (frame.empty()) {
//       std::cout << "Failed to capture a frame." << std::endl;
//       break;
//     }

//     cv::imshow("Video", frame); // Display the frame

//     // Exit the loop if the 'Esc' key is pressed
//     if (cv::waitKey(1) == 27)
//       break;
//   }

//   cv::destroyAllWindows(); // Close all windows

//   return 0;
// }