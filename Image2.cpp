#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdlib.h>


using namespace cv;
using namespace std;

static string filepath_image ="img/";


int main(int argc, char* argv[])
{
    cv::Mat image2 = cv::imread(filepath_image+"Image2.png",IMREAD_GRAYSCALE);

    if (image2.empty()) {
        std::cout << "Input image not found at '" << filepath_image << "'\n";
        return 1;
    }

    // Good of Image 2 Median blur with 11 kernel size
    cv::Mat medianImage2;
    medianBlur(image2,medianImage2,11);
    cv::imshow("Median blurred image", medianImage2);
    imwrite(filepath_image+"restored_Image2.png",medianImage2);


    cv::imshow("Original Image", image2);


    while (cv::waitKey() != 24);

    return 0;
}
