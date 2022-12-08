#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace cv;

static string filepath_image ="img/";


int main(int argc, char* argv[])
{
    cv::Mat image1 = cv::imread(filepath_image+"Image1.png",IMREAD_GRAYSCALE);

    if (image1.empty()) {

        std::cout << "Input image1 not found at '" << filepath_image << "'\n";

        return 1;

    }

    // Dilate Filter
    cv::Mat maxImage1;

    dilate(image1,maxImage1,cv::Mat(),cv::Point(-1,1),2,cv::BORDER_CONSTANT,1);
    cv::imshow ("Maximized image", maxImage1);
    imwrite(filepath_image+"restored_Image1.png",maxImage1);

    cv::imshow("Original Image", image1);


    while (cv::waitKey() != 24);

    return 0;
}
