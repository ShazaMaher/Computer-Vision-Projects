#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdlib.h>

using namespace std;
using namespace cv;

static string filepath_image ="img/";



cv::Mat bandreject_Butterworth_filter (float W, int m , int d0, cv::Size size)
{
    cv::Mat_<cv::Vec2f> hlBf(size);

    cv::Point2f c = cv::Point2f(size) / 2;

    for (int i = 0; i < size.height; ++i)
        {
            for (int j = 0; j < size.width; ++j)
                {

                    float d = std::sqrt((i - c.y ) * (i - c.y) + (j - c.x) * (j - c.x));


                    if (std::abs(d) < 1.e-9f)
                        hlBf(i, j)[0] = 0;
                    else
                        hlBf(i, j)[0] = 1 / (1 + std::pow(W / (d*d - d0*d0), 2 * m));


                     hlBf(i, j)[1] = 0;
                 }
        }

    return hlBf;
}



void dftshift(cv::Mat& mag)
{
    int cx = mag.cols / 2;
    int cy = mag.rows / 2;

    cv::Mat temp;
    cv::Mat q0(mag, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(mag, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(mag, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(mag, cv::Rect(cx, cy, cx, cy));

    q0.copyTo(temp);
    q3.copyTo(q0);
    temp.copyTo(q3);

    q1.copyTo(temp);
    q2.copyTo(q1);
    temp.copyTo(q2);
}


cv::Mat takedft(cv::Mat& source, cv::Mat& destination)
{
    cv::Mat imageComplex[2]= {source, cv::Mat::zeros(source.size(), CV_32F)};

    cv::Mat dftready;

    merge (imageComplex,2,dftready);

    cv::Mat dftofimage;

    dft(dftready, dftofimage);

    destination = dftofimage;

    return destination;

}

cv::Mat showdft(cv::Mat& source, cv::Mat& dest)
{
    cv::Mat splitArray[2] = {cv::Mat::zeros(source.size(), CV_32F) ,cv::Mat::zeros(source.size(), CV_32F)};

    split(source, splitArray);

    cv::Mat dftmagnitude;

    magnitude (splitArray[0],splitArray[1], dftmagnitude);

    dftmagnitude += cv::Scalar::all(1);

    log(dftmagnitude,dftmagnitude);

    normalize (dftmagnitude, dftmagnitude, 0, 1, CV_MINMAX);

    dest = dftmagnitude;

    return dest;

}







int main(int argc, char* argv[])
{
    cv::Mat image4_2 = cv::imread(filepath_image+"Image4_2.png",IMREAD_GRAYSCALE);

    if (image4_2.empty()) {

        std::cout << "Input image1 not found at '" << filepath_image << "'\n";

        return 1;

    }


    // Expand the image to an optimal size.
    cv::Mat padded;
    int opt_rows = cv::getOptimalDFTSize(image4_2.rows * 2 - 1);
    int opt_cols = cv::getOptimalDFTSize(image4_2.cols * 2 - 1);
    cv::copyMakeBorder(image4_2, padded, 0, opt_rows - image4_2.rows, 0, opt_cols - image4_2.cols,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // Make place for both the real and complex values by merging planes into a
    // cv::Mat with 2 channels.
    // Use float element type because frequency domain ranges are large.
    cv::Mat planes[] = {
        cv::Mat_<float>(padded),
        cv::Mat_<float>::zeros(padded.size())
    };
    cv::Mat complex;
    cv::merge(planes, 2, complex);

    // Compute DFT of image
    cv::dft(complex, complex);

    // Shift quadrants to center
    dftshift(complex);



    cv::Mat filter2;


    filter2 = bandreject_Butterworth_filter(320000, 4, 830, complex.size());
    cv::mulSpectrums(complex, filter2, complex, 0);


     dftshift(complex);

     // Compute inverse DFT
     cv::Mat filteredImage4_2;
     cv::idft(complex, filteredImage4_2, (cv::DFT_SCALE | cv::DFT_REAL_OUTPUT));

     // Crop image (remove padded borders)
     filteredImage4_2 = cv::Mat(filteredImage4_2, cv::Rect(cv::Point(0, 0), image4_2.size()));

     cv::normalize(filteredImage4_2, filteredImage4_2, 0, 1, cv::NORM_MINMAX);
     cv::imshow("Filtered image", filteredImage4_2);




     cv::Mat filter_planes[2];
     cv::split(filter2, filter_planes);
     cv::normalize(filter_planes[0], filter_planes[0], 0, 1, cv::NORM_MINMAX);
     cv::imshow("Filter", filter_planes[0]);



    cv::imshow("Original Image", image4_2);


    while (cv::waitKey() != 24);

    return 0;
}
