#include <opencv2/opencv.hpp>
#include <black_fly_tools/black_fly_lib.hpp>
#include "Spinnaker.h"
#include "SpinVideo.h"


int main(){

    // Create a BlackFly camera object
    bfc::BlackFlyCamera camera;
    std::cout << "Camera object created" << std::endl;

    // Set exposure to auto
    camera.set_auto_exposure("Continuous");
    // camera.set_exposure_time(1000.0);

    // camera.set_auto_gain("Continuous");
    camera.set_gain(15.0);
    // Begin aquiring frames
    camera.begin_acquisition();

    // Get the first frame to determine the frame size
    cv::Mat frame = camera.get_frame();
    cv::Size frameSize = frame.size(); //frameSize(2048, 1536); // Set the frame size to your desired dimensions

    // Create a window to display the video
    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video", 800, 600);

    bool flag = true; // Flag to exit the loop
    while (flag)
    {
        frame = camera.get_frame(); // Get a frame from the camera
        cv::imshow("Video", frame); // Display the frame

        // Exit the loop if the 'Esc' key is pressed
        if (cv::waitKey(1) == 27)
            flag = false;
    }
    std::cout << "Stream Ended" << std::endl;

    cv::destroyAllWindows();
    std::cout << "Writer Released" << std::endl;

    return 0;
}
