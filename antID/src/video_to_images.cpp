#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>
#include<tuple> 
#include <iostream>

cv::Mat apply_mask(cv::Mat frame, cv::Mat mask){
    // Convert mask to 3 channel image
    cv::Mat masked_image;
    cv::cvtColor(mask, masked_image, cv::COLOR_GRAY2BGR);

    // Apply mask to frame
    cv::bitwise_and(frame, masked_image, masked_image);
    return masked_image;
}


int main(){
    // WILL HAVE TO LOOP THROUGH ALL VIDEOS LATER

    // Create background subtractor
    cv::Ptr<cv::BackgroundSubtractor> pBackSub;
    pBackSub = cv::createBackgroundSubtractorKNN();

    // Open video
    cv::VideoCapture cap("ant_100.avi");

    // Check if video opened successfully
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }
    else{
        std::cout << "Video opened successfully" << std::endl;
    }

    // Create kernel for morphological closing
    auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    // Loop through video
    while(1){
        // Get frame from video
        cv::Mat frame;
        cap >> frame;

        // Blur frame
        cv::GaussianBlur(frame, frame, cv::Size(25, 25), 10);

        // Apply background subtraction
        cv::Mat fg_mask;
        pBackSub->apply(frame, fg_mask);

        // Threshold mask
        cv::threshold(fg_mask, fg_mask, 250, 255, cv::THRESH_BINARY);

        // Apply morphological opening
        cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_OPEN, kernel);

        // Apply morphological closing
        cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_CLOSE, kernel);

        // Apply Canny edge detection to mask
        cv::Canny(fg_mask, fg_mask, 150, 200);

        // // Get contours
        // std::vector<std::vector<cv::Point>> contours;
        // cv::findContours(fg_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Get all pixels in image that are not black
        std::vector<cv::Point> points;
        cv::findNonZero(fg_mask, points);

        // Get bounding box around ant and filter if not full ant
        if (points.size() > 0){
            // Get leftmost, rightmost, topmost, and bottommost points of ant
            auto leftmost_x = points.at(0).x;
            auto rightmost_x = points.back().x;
            auto topmost_y = points.at(0).y;
            auto bottommost_y = points.back().y;

            // Get area of bounding box
            auto bounding_box_area =  (rightmost_x - leftmost_x) * (bottommost_y - topmost_y);

            // Get image area
            auto image_area = fg_mask.rows * fg_mask.cols;

            // Get percent of bounding box area to image area
            auto percent_area = bounding_box_area / image_area;

            // Filter if bounding box area is less than 22.35%  and greater than 70% of image area
            if (percent_area < 0.2235 || percent_area > 0.7){
                // Draw bounding box around ant
                cv::rectangle(frame, cv::Point(leftmost_x, topmost_y), cv::Point(rightmost_x, bottommost_y), cv::Scalar(0, 255, 0), 2);
            }
        }


        cv::Mat masked_image = apply_mask(frame, fg_mask);

        // Display frame
        cv::imshow("Frame", frame);

        // Press  ESC on keyboard to exit
        char c=(char)cv::waitKey(25);
        if(c==27){
            break;
        }
    }
    return 0;
}