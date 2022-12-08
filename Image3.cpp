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
    cv::Mat image3 = cv::imread(filepath_image+"Image3.png",IMREAD_GRAYSCALE);

    if (image3.empty()) {

        std::cout << "Input image1 not found at '" << filepath_image << "'\n";

        return 1;

    }

    //first solution using Bilateral filter
    cv::Mat BilateralImage3;
    cv::bilateralFilter(image3,BilateralImage3,7,60,60,cv::BORDER_DEFAULT);
    cv::imshow("Bilateral blurred image", BilateralImage3);
    imwrite(filepath_image+"restored_Image3_Bilateral.png",BilateralImage3);


    cv::imshow("Original Image", image3);




    //second solution using Gaussian filter
    // Output image of Image 3 after applying Gaussian blur with 13X13 kernel size
    cv::Mat GaussianImage3;
    cv::GaussianBlur(image3,GaussianImage3,cv::Size(13,13),0,cv::BORDER_CONSTANT);
    cv::imshow("Gaussian blurred image", GaussianImage3);
    imwrite(filepath_image+"restored_Image3_Gaussian.png",GaussianImage3);




    while (cv::waitKey() != 24);

    return 0;
}
