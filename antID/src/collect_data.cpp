#include <iostream>
#include <black_fly_tools/black_fly_lib.hpp>
#include "turtlelib/rigid2d.hpp"

using namespace bfc;

int main(){
    // Get ID number of ant to start pause_flaging
    std::cout << "Enter starting ant ID number:";
    int curr_id;
    std::cin >> curr_id;

    // Create camera object
    BlackFlyCamera camera;

    // Set exposure time
    camera.set_exposure_time(1500);

    // Enable and set gamma correction
    camera.enable_gamma(true);
    camera.set_gamma(0.5);

    // Manually set gain
    camera.set_gain(20.0);

    // Begin capturing images
    camera.begin_acquisition();

    // Set frames per second
    double fps = 55;

    // Get frame size
    cv::Mat frame = camera.get_frame();
    cv::Size frameSize = frame.size(); 

    // Create window to display video
    cv::namedWindow("Live Feed", cv::WINDOW_NORMAL);
    cv::resizeWindow("Live Feed", 800, 600);

    // Create video writer object
    std::string outputFile = "../../labeled_vids/ant_" + std::to_string(curr_id) + ".avi";
    cv::VideoWriter writer(outputFile, cv::VideoWriter::fourcc('M','J','P','G'), fps, frameSize);

    // Check if the video writer is open
    if (!writer.isOpened())
    {
        // Error handling if the video writer cannot be opened
        std::cerr << "Failed to open the output video file for writing" << std::endl;
        return -1;
    }

    bool end_flag = true; // end_flag to end pause_flaging
    bool pause_flag = true; // end_flag to pause pause_flaging
    std::cout << "Press space to pause" << std::endl;

    while (end_flag)
    {
        if (pause_flag){
            // Get frame from camera
            frame = camera.get_frame();
            cv::imshow("Live Feed", frame); // Display the frame
            writer.write(frame);
        }

        // Press Space to pause pause_flaging
        if (cv::waitKey(1) == 32){
            std::cout << "You have paused" << std::endl;
            std::cout << "Press:" << std::endl;
            std::cout << "    'q' to quit session" << std::endl;
            std::cout << "    'n' to start new video" << std::endl;
            std::cout << "    'c' to continue" << std::endl;
            std::cout << "Enter now: ";
            std::string input;
            std::cin >> input;

            if (input == "q"){
                std::cout << "You have ended the session" << std::endl;
                end_flag = false;
            }
            else if (input == "n"){
                std::cout << "You have created a new video" << std::endl;

                // Increment ant ID number and update output file name
                curr_id++;
                outputFile = "ant_" + std::to_string(curr_id) + ".avi";

                // Release current video writer and create new one
                writer.release();
                writer = cv::VideoWriter(outputFile, cv::VideoWriter::fourcc('M','J','P','G'), fps, frameSize);
                pause_flag = true;
            }
            else if (input == "c"){
                continue;
            }
            else{
                std::cout << "Invalid input" << std::endl;
            }
        }
    }

    std::cout << "pause_flaging Ended" << std::endl;
    writer.release();

    cv::destroyAllWindows();
    std::cout << "Writer Released" << std::endl;    return 0;
    return 0;
}