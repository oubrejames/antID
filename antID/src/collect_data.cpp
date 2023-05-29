#include <iostream>
#include <black_fly_tools/black_fly_lib.hpp>
#include "turtlelib/rigid2d.hpp"

using namespace bfc;

int main(){
    std::cout << "Enter starting ant ID number:";
    int curr_id;
    std::cin >> curr_id;

    // Create camera object
    BlackFlyCamera camera;

    // Set exposure time
    camera.set_exposure_time(5000);

    // Begin capturing images
    camera.begin_acquisition();

    // Set frames per second
    double fps = 20;

    // Get frame size
    cv::Mat frame = camera.get_frame();
    cv::Size frameSize = frame.size(); 

    // Create window to display video
    cv::namedWindow("Live Feed", cv::WINDOW_NORMAL);
    cv::resizeWindow("Live Feed", 800, 600);

    // Create video writer object
    std::string outputFile = "ant_" + std::to_string(curr_id) + ".avi";
    cv::VideoWriter writer(outputFile, cv::VideoWriter::fourcc('M','J','P','G'), fps, frameSize);

    // Open the video file for writing
    // writer.open(outputFile, fourcc, fps, frameSize);
    if (!writer.isOpened())
    {
        // Error handling if the video writer cannot be opened
        std::cerr << "Failed to open the output video file for writing" << std::endl;
        return -1;
    }

    bool flag = true;
    bool record = true;
    while (flag)
    {
        if (record){
            frame = camera.get_frame();
            // cap >> frame;
            cv::imshow("Live Feed", frame); // Display the frame
            writer.write(frame);
        }

        // Press Space to pause recording
        if (cv::waitKey(1) == 32){
            std::cout << "You have paused recording" << std::endl;
            std::cout << "Press:" << std::endl;
            std::cout << "    'q' to quit session" << std::endl;
            std::cout << "    'n' to start new video" << std::endl;
            std::cout << "    'c' to continue" << std::endl;
            std::cout << "Enter now: ";
            std::string input;
            std::cin >> input;

            if (input == "q"){
                std::cout << "You have ended the session" << std::endl;
                flag = false;
            }
            else if (input == "n"){
                std::cout << "You have created a new video" << std::endl;
                curr_id++;
                outputFile = "ant_" + std::to_string(curr_id) + ".avi";
                writer.release();
                // frame = camera.get_frame();
                writer = cv::VideoWriter(outputFile, cv::VideoWriter::fourcc('M','J','P','G'), fps, frameSize);
                record = true;
            }
            else if (input == "c"){
                continue;
            }
            else{
                std::cout << "Invalid input" << std::endl;
            }
            // flag = false;
        }

        // // Stop recording but don't exit the loop if the 's' key is pressed
        // if (cv::waitKey(1) == 115){
        //     std::cout << "You have stopped recording" << std::endl;
        //     record = false;
        // }

        // // Create a new video if the 'n' key is pressed
        // if (cv::waitKey(1) == 110){
        //     std::cout << "You have created a new video" << std::endl;
        //     curr_id++;
        //     outputFile = "ant_" + std::to_string(curr_id) + ".avi";
        //     writer.release();
        //     writer = cv::VideoWriter(outputFile, cv::VideoWriter::fourcc('M','J','P','G'), fps, frameSize);
        //     record = true;
        // }
    }
    std::cout << "Recording Ended" << std::endl;
    writer.release();

    cv::destroyAllWindows();
    std::cout << "Writer Released" << std::endl;    return 0;
    return 0;
}