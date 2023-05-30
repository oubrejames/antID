#include <opencv2/opencv.hpp>
#include <black_fly_tools/black_fly_lib.hpp>
#include "Spinnaker.h"
#include "SpinVideo.h"


int main(){

    // Create a BlackFly camera object
    bfc::BlackFlyCamera camera;
    std::cout << "Camera object created" << std::endl;

    // Set exposure time
    camera.set_exposure_time(500.0);

    // Begin aquiring frames
    camera.begin_acquisition();

    // Set frame rate for CV video writer
    double fps = 20;
    std::cout << "FPS: " << fps << std::endl;

    // Get the first frame to determine the frame size
    cv::Mat frame = camera.get_frame();
    cv::Size frameSize = frame.size();

    // Create a window to display the video
    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video", 800, 600);

    // Create a video writer object
    cv::VideoWriter writer("output_video.avi", cv::VideoWriter::fourcc('M','J','P','G'), fps, frameSize);

    // Check if the video writer was successfully opened
    if (!writer.isOpened())
    {
        // Error handling if the video writer cannot be opened
        std::cerr << "Failed to open the output video file for writing" << std::endl;
        return -1;
    }

    // Loop to display and record frames
    bool flag = true; // Flag to exit the loop
    while (flag)
    {
        frame = camera.get_frame(); // Get a frame from the camera
        cv::imshow("Video", frame); // Display the frame

        writer.write(frame);  // Write the frame to the video file

        // Exit the loop if the 'Esc' key is pressed
        if (cv::waitKey(1) == 27)
            flag = false;
    }
    std::cout << "Recording Ended" << std::endl;

    writer.release();
    cv::destroyAllWindows();
    std::cout << "Writer Released" << std::endl;

    return 0;
}
