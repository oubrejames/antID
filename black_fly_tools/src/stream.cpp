#include <opencv2/opencv.hpp>
#include <black_fly_tools/black_fly_lib.hpp>
#include "Spinnaker.h"
#include "SpinVideo.h"


int main(){

    bfc::BlackFlyCamera camera;
    std::cout << "Camera object created" << std::endl;
    camera.set_auto_exposure("Continuous");
    camera.begin_acquisition();

    cv::Mat frame = camera.get_frame();
    cv::Size frameSize = frame.size(); //frameSize(2048, 1536); // Set the frame size to your desired dimensions

    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video", 800, 600);

    bool flag = true;
    while (flag)
    {
        frame = camera.get_frame();
        // cap >> frame;
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
