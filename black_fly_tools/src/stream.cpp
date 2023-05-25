#include <opencv2/opencv.hpp>
#include <black_fly_lib.hpp>
#include "Spinnaker.h"

int main(){
    bfc::BlackFlyCamera camera;
    camera.initialize_camera();
    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video", 800, 600);

    while (true) {
        cv::Mat frame = camera.get_frame();
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