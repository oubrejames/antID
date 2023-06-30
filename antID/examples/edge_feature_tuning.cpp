#include <gtk/gtk.h>
#include <iostream>
#include <opencv2/opencv.hpp>


cv::Mat makeCanvas(std::vector<cv::Mat>& vecMat, int windowHeight, int nRows) {
        int N = vecMat.size();
        nRows  = nRows > N ? N : nRows; 
        int edgeThickness = 10;
        int imagesPerRow = ceil(double(N) / nRows);
        int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;
        int maxRowLength = 0;

        std::vector<int> resizeWidth;
        for (int i = 0; i < N;) {
                int thisRowLen = 0;
                for (int k = 0; k < imagesPerRow; k++) {
                        double aspectRatio = double(vecMat[i].cols) / vecMat[i].rows;
                        int temp = int( ceil(resizeHeight * aspectRatio));
                        resizeWidth.push_back(temp);
                        thisRowLen += temp;
                        if (++i == N) break;
                }
                if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength) {
                        maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
                }
        }
        int windowWidth = maxRowLength;
        cv::Mat canvasImage(windowHeight, windowWidth, CV_8UC3, cv::Scalar(0, 0, 0));

        for (int k = 0, i = 0; i < nRows; i++) {
                int y = i * resizeHeight + (i + 1) * edgeThickness;
                int x_end = edgeThickness;
                for (int j = 0; j < imagesPerRow && k < N; k++, j++) {
                        int x = x_end;
                        cv::Rect roi(x, y, resizeWidth[k], resizeHeight);
                        cv::Size s = canvasImage(roi).size();
                        // change the number of channels to three
                        cv::Mat target_ROI(s, CV_8UC3);
                        if (vecMat[k].channels() != canvasImage.channels()) {
                            if (vecMat[k].channels() == 1) {
                                cv::cvtColor(vecMat[k], target_ROI, cv::COLOR_GRAY2BGR);
                            }
                        } else {             
                            vecMat[k].copyTo(target_ROI);
                        }
                        cv::resize(target_ROI, target_ROI, s);
                        if (target_ROI.type() != canvasImage.type()) {
                            target_ROI.convertTo(target_ROI, canvasImage.type());
                        }
                        target_ROI.copyTo(canvasImage(roi));
                        x_end += resizeWidth[k] + edgeThickness;
                }
        }
        return canvasImage;
}



const int alpha_slider_max = 100;
int alpha_slider;
int thresh2;
int thresh1;
cv::Mat src1;
cv::Mat edges;
int blur_size=1;
int sigma=0;
int blockSize;

static void on_blockSize_trackbar( int val, void* ){
    if (val % 2==0){
        blockSize = val+1;
        cv::setTrackbarPos("Blur ssize", "Linear Blend", blockSize);
    }
    else{
        blockSize=val;
        cv::setTrackbarPos("Blur ssize", "  Linear Blend", blockSize);
    }

    if(blockSize < 1){
        blockSize = 1;
    }
}

static void on_trackbar( int, void* )
{
    cv::Mat blr;
    cv::GaussianBlur(src1, blr, cv::Size(blur_size, blur_size), sigma);
   cv::Canny(blr, edges, thresh1, thresh2);
   std::vector<cv::Mat> canvas;
    canvas.push_back(src1);
    canvas.push_back(edges);
    canvas.push_back(blr);

    cv::Mat canvas_frame = makeCanvas(canvas, 500, 1);
   cv::imshow( "Linear Blend", canvas_frame );
}

int main( void )
{
   src1 = cv::imread( "../../labeled_images/ant_1/ant_12_im_12.jpg");

   int threshold1 = 0;
   int threshold2 = 0;
   cv::namedWindow("Linear Blend", cv::WINDOW_AUTOSIZE); // Create Window
   char TrackbarName[50];
   sprintf( TrackbarName, "Alpha x %d", alpha_slider_max );
   cv::createTrackbar( "Canny Lower Threshold", "Linear Blend", &thresh1, 500, on_trackbar );
   cv::createTrackbar( "Canny Upper Threshold", "Linear Blend", &thresh2, 500, on_trackbar );
   cv::createTrackbar( "Blur sigma", "Linear Blend", &sigma, 500, on_trackbar );
   cv::createTrackbar( "Blur ssize", "Linear Blend", &blur_size, 500, on_blockSize_trackbar );

   on_trackbar(threshold1, 0 );
   cv::waitKey(0);
   return 0;
}