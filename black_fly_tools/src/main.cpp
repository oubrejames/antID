#include <opencv2/opencv.hpp>
#include <black_fly_lib.hpp>

int main() {

    // Create a BlackFlyCamera object
    bfc::BlackFlyCamera cam;

    // Set the camera to continuous acquisition mode
    cam.set_continuous_acquisition();



  while (true) {
    cv::Mat frame;
    frame = cam.get_frame(); // Read a new frame from the camera

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
