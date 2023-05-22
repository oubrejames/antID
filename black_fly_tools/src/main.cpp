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

    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video", 800, 600);

  while (true) {
    pResultImage = pCam->GetNextImage(1000);
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
