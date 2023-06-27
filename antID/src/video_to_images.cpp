#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>
#include<tuple> 
#include <iostream>
#include <algorithm>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>

#define IMAGE_HEIGHT 1536
#define IMAGE_WIDTH 2048
#define PATH_TO_VIDEOS "../../labeled_vids"

//////////////////////////////////////////
// REMEMBER TO TUNE BOUNDING BOX % AREA //
//////////////////////////////////////////

bool detect_full_body(std::vector<cv::Point> points, cv::Mat* frame){
    // Get bounding box around ant and filter if not full ant
    if (points.size() > 0){
        // Get leftmost, rightmost, topmost, and bottommost points of ant
        auto leftmost_x_point = std::min_element(points.begin(), points.end(), [](cv::Point a, cv::Point b){return a.x < b.x;});
        auto leftmost_x = leftmost_x_point->x;

        auto rightmost_x_point = std::max_element(points.begin(), points.end(), [](cv::Point a, cv::Point b){return a.x < b.x;});
        auto rightmost_x = rightmost_x_point->x;

        auto topmost_y_point = std::min_element(points.begin(), points.end(), [](cv::Point a, cv::Point b){return a.y < b.y;});
        auto topmost_y = topmost_y_point->y;

        auto bottommost_y_point = std::max_element(points.begin(), points.end(), [](cv::Point a, cv::Point b){return a.y < b.y;});
        auto bottommost_y = bottommost_y_point->y;

        // Get area of bounding box
        auto bounding_box_area =  (rightmost_x - leftmost_x) * (bottommost_y - topmost_y);

        // Get image area
        auto image_area = IMAGE_HEIGHT * IMAGE_WIDTH;

        // Get percent of bounding box area to image area
        auto percent_area = static_cast<float>(bounding_box_area) / static_cast<float>(image_area);

        // Filter if bounding box area is less than 22.35%  and greater than 70% of image area
        if (percent_area > 0.26 && percent_area < 0.8){
            // Draw bounding box around ant
            cv::rectangle(*frame, cv::Point(leftmost_x, topmost_y), cv::Point(rightmost_x, bottommost_y), cv::Scalar(0, 255, 0), 5);
            return true;
        }
        else{
            return false;
        }
    }
}

int main(){
    // Loop through all the videos in the labeled videos directory
    const std::filesystem::path labeled_videos{PATH_TO_VIDEOS};

    for (auto const& video_path : std::filesystem::directory_iterator{labeled_videos}) 
    {
        // Open video
        cv::VideoCapture cap(video_path.path());

        std::cout << "Opening video : " << video_path.path() << std::endl;

        auto ant_id = video_path.path().stem().string().back();
        std::cout << "Ant ID : " << ant_id << typeid(ant_id).name()<< std::endl;

        // Check if video opened successfully
        if(!cap.isOpened()){
            std::cout << "Error opening video stream or file" << std::endl;
            return -1;
        }
        else{
            std::cout << "Video opened successfully" << std::endl;
        }

        // Create background subtractor
        cv::Ptr<cv::BackgroundSubtractor> pBackSub;
        pBackSub = cv::createBackgroundSubtractorKNN();

        // Create kernel for morphological closing
        auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

        // Make directory for labeled images
        std::string dir_labeled_imgs = "../../labeled_images/ant_";

        dir_labeled_imgs.push_back(ant_id);
        std::filesystem::create_directory(dir_labeled_imgs, labeled_videos);
        std::cout << "Directory created : " << dir_labeled_imgs << std::endl;

        // Loop through video
        int img_count = 0;
        while(1){
            // Get frame from video
            cv::Mat frame;
            cap >> frame;

            if (frame.empty()){
                break;
            }

            // Blur frame
            cv::Mat blurred_frame;
            cv::GaussianBlur(frame, blurred_frame, cv::Size(25, 25), 10);

            // Apply background subtraction
            cv::Mat fg_mask;
            pBackSub->apply(blurred_frame, fg_mask);

            // Threshold mask
            cv::threshold(fg_mask, fg_mask, 250, 255, cv::THRESH_BINARY);

            // Apply morphological opening
            cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_OPEN, kernel);

            // Apply morphological closing
            cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_CLOSE, kernel);

            // Get all pixels in image that are not black
            std::vector<cv::Point> white_points;
            cv::findNonZero(fg_mask, white_points);

            // If a full ant is detected, save the frame 
            if (detect_full_body(white_points, &frame)){
                // Save frame
                cv::imwrite(dir_labeled_imgs + "/ant_" + std::to_string(img_count) + "_im_" + std::to_string(img_count) + ".jpg", frame);
                img_count++;
            }

            // Create and resize windows
            cv::namedWindow("Frame", cv::WINDOW_NORMAL);
            cv::resizeWindow("Frame", 800, 600);

            // Display frame
            cv::imshow("Frame", frame);

            // Press  ESC on keyboard to exit
            char c=(char)cv::waitKey(25);
            if(c==27){
                break;
            }
        }
    std::cout << "Processed video : " << video_path.path() << std::endl << std::endl;
    }
    return 0;
}