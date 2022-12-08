#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdlib.h>


using namespace std;
using namespace cv;



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





cv::Mat histogram(const cv::Mat image)
{
  assert(image.type() == CV_8UC1);

    cv::Mat None;
    cv::Mat hist;
    cv::calcHist(std::vector<cv::Mat>{image},{0}, None ,hist, {256}, {0, 256});
    return hist;
}



cv::Mat draw_the_histogram_of_image(const cv::Mat hist)
{
    int nbins = hist.rows;
    double max = 0;
    cv::minMaxLoc(hist, nullptr, &max);
    cv::Mat image(nbins, nbins, CV_8UC1, cv::Scalar(255));

    for (int i = 0; i < nbins; i++)
        {
            double h = nbins * (hist.at<float>(i) / max);
            cv::line(image, cv::Point(i, nbins), cv::Point(i, nbins - h), cv::Scalar::all(0));
        }

    return image;
}



int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv,
        "{help   |            | print this message}"
        "{@image | img/Image1.png | image path}"
        "{filter |            | toggle to high-pass filter the input image}"
    );

    if (parser.has("help"))
        {
            parser.printMessage();
            return 0;
        }

    // Loading of image as grayscale
    std::string filepath = parser.get<std::string>("@image");
    cv::Mat image = cv::imread(filepath, cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cout << "Input image not found at '" << filepath << "'\n";
        return 1;
    }


    cv:: Mat hist;
    hist = histogram(image);
    cv:: imshow("Histogram of Image", draw_the_histogram_of_image(hist));


    int  x1=10;
    int y1=10;
    int width1 = 100;
    int height1=100;
  cv::Rect croppedrect1 = Rect (x1,y1,width1,height1);
  cv::Mat cropped1 = image(croppedrect1);
  cv::imshow ("Cropped Image1", cropped1);
  cv::Mat hist_of_cropped1= histogram(cropped1);
  cv::imshow("Histogram of cropped1", draw_the_histogram_of_image(hist_of_cropped1) );




  cv:: Mat imagefloat;

  image.convertTo(imagefloat, CV_32F, 1.0/255.0);


  cv::Mat dftofimage;

  takedft(imagefloat, dftofimage);

  cv::Mat dftmagnitude;

  showdft(dftofimage, dftmagnitude);


  dftshift(dftmagnitude);


  imshow("Shifted DFT of Image", dftmagnitude);


    cv::imshow("Original Image", image);


    while (cv::waitKey() != 24);

    return 0;
}
