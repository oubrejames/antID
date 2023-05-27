#include <opencv2/opencv.hpp>
#include <black_fly_tools/black_fly_lib.hpp>
#include "Spinnaker.h"
#include "SpinVideo.h"


int main(){

    bfc::BlackFlyCamera camera;
    std::cout << "Camera object created" << std::endl;
    camera.set_exposure_time(250.0);
    camera.begin_acquisition();

    double fps = 10; //camera.fps; // Frames per second
    std::cout << "FPS: " << fps << std::endl;

    cv::Mat frame = camera.get_frame();
    cv::Size frameSize = frame.size(); //frameSize(2048, 1536); // Set the frame size to your desired dimensions

    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video", 800, 600);
    cv::VideoWriter writer("output_video.avi", cv::VideoWriter::fourcc('M','J','P','G'), fps, frameSize);

    // Open the video file for writing
    // writer.open(outputFile, fourcc, fps, frameSize);
    if (!writer.isOpened())
    {
        // Error handling if the video writer cannot be opened
        std::cerr << "Failed to open the output video file for writing" << std::endl;
        return -1;
    }

    bool flag = true;
    while (flag)
    {
        frame = camera.get_frame();
        // cap >> frame;
        cv::imshow("Video", frame); // Display the frame

        // int width = frame.cols;
        // int height = frame.rows;

        // std::cout << "Frame dimensions: " << width << " x " << height << std::endl;

        writer.write(frame);

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
