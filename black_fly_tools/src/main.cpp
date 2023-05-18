#include <opencv2/opencv.hpp>
#include <black_fly_lib.hpp>

int main() {
  cv::VideoCapture cap(0); // Open the default camera (index 0)

  if (!cap.isOpened()) {
    std::cout << "Failed to open the camera." << std::endl;
    return -1;
  }

  while (true) {
    cv::Mat frame;
    cap.read(frame); // Read a new frame from the camera

    if (frame.empty()) {
      std::cout << "Failed to capture a frame." << std::endl;
      break;
    }

    cv::imshow("Video", frame); // Display the frame

    // Exit the loop if the 'Esc' key is pressed
    if (cv::waitKey(1) == 27)
      break;
  }

  cap.release(); // Release the video capture object
  cv::destroyAllWindows(); // Close all windows

  return 0;
}
