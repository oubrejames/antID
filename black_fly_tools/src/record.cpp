#include <opencv2/opencv.hpp>
#include <black_fly_lib.hpp>
#include "Spinnaker.h"
#include "SpinVideo.h"


int main(){
    std::cout << "yaga" << std::endl;

    bfc::BlackFlyCamera camera;
    std::cout << "Camera object created" << std::endl;

    camera.initialize_camera();
    std::cout << "Camera initialized" << std::endl;

    // cv::VideoWriter writer;
    // Define the output file name, codec, frames per second (fps), and frame size
    // std::string outputFile = "output.avi";
    // int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');; // Codec for AVI format
    double fps = 55; //camera.fps; // Frames per second
    std::cout << "FPS: " << fps << std::endl;

    /////////////
    cv::Mat frame = camera.get_frame();
    cv::Size frameSize = frame.size(); //frameSize(2048, 1536); // Set the frame size to your desired dimensions
    /////////////

    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video", 800, 600);
    cv::VideoWriter writer("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), fps, frameSize);

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

// int main(){
//     SpinVideo video;
//     Spinnaker::Video::H264Option option;
//     std::string videoFilename = "testy";
//     option.frameRate = 50;
//     option.bitrate = 1000000;
//     option.height = static_cast<unsigned int>(images[0]->GetHeight());
//     option.width = static_cast<unsigned int>(images[0]->GetWidth());
//     video.Open(videoFilename, option)

//     // Construct and save video
//     //
//     // *** NOTES ***
//     // Although the video file has been opened, images must be individually
//     // appended in order to construct the video.
//     //
//     std::cout << "Appending " << images.size() << " images to video file: " << videoFilename << ".avi... " << std::endl
//             << std::endl;
//     for (unsigned int imageCnt = 0; imageCnt < images.size(); imageCnt++)
//     {
//         video.Append(images[imageCnt]);
//         std::cout << "\tAppended image " << imageCnt << "..." << std::endl;
//     }
//     //
//     // Close video file
//     //
//     // *** NOTES ***
//     // Once all images have been appended, it is important to close the
//     // video file. Notice that once an video file has been closed, no more
//     // images can be added.
//     //
//     video.Close();
//     std::cout << std::endl << "Video saved at " << videoFilename << ".avi" << std::endl << std::endl;
//     return 0;
// }