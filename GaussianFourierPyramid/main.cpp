#include "LLF.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;

#pragma region lib_link
#define CV_LIB_PREFIX "opencv_"

#define CV_LIB_VERSION CVAUX_STR(CV_MAJOR_VERSION)\
    CVAUX_STR(CV_MINOR_VERSION)\
    CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define CV_LIB_SUFFIX CV_LIB_VERSION "d.lib"
#else
#define CV_LIB_SUFFIX CV_LIB_VERSION ".lib"
#endif

#define CV_LIBRARY(lib_name) CV_LIB_PREFIX CVAUX_STR(lib_name) CV_LIB_SUFFIX
#pragma comment(lib, CV_LIBRARY(core))
#pragma comment(lib, CV_LIBRARY(highgui))
#pragma comment(lib, CV_LIBRARY(imgcodecs))
#pragma comment(lib, CV_LIBRARY(imgproc))
#pragma endregion

//alpha blending comparison between src1 and src2 by GUI
void compare(const string wname, const Mat& src1, const Mat& src2)
{
	namedWindow(wname);
	int a = 0; createTrackbar("alpha", wname, &a, 100);
	int key = 0;
	Mat show;
	while (key != 'q')
	{
		addWeighted(src1, double(a) * 0.01, src2, double(100 - a) * 0.01, 0.0, show);
 		imshow(wname, show);
		key = waitKey(1);
	}
}

int dbt()
{
	//load image
	Mat src = imread("s_0020.png", cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	src.convertTo(src, CV_32F);
	cv::normalize(src, src, 0., 255., cv::NormTypes::NORM_MINMAX);
	cv::Mat vMask;
	cv::compare(src, 0., vMask, cv::CmpTypes::CMP_GT);


	//destination image
	Mat destFastLLF, destFourierLLF, destFastLLFAaptive, destFourierLLFAaptive;

	//parameter setting
	const float sigma = 30.f;
	const float boost = 3.f;
	const int level = 6;
	const int order = 8;

	//create instance
	FastLLF llf;
	GaussianFourierLLF gfllf;

	//parameter fix filter
	llf.filter(src, destFastLLF, order * 2, sigma, boost, level);//order*2: FourierLLF requires double pyramids due to cos and sin pyramids; thus we double the order to adjust the number of pyramids.
	gfllf.filter(src, destFourierLLF, order, sigma, boost, level);

	//parameter adaptive filter
	//generate parameter maps (circle shape)
	Mat sigmaMap(src.size(), CV_32F);
	sigmaMap.setTo(1.f);
	sigmaMap.setTo(sigma,vMask);

	cv::Mat boostGain;
	src.copyTo(boostGain);
	cv::Mat d;
	d = (boostGain - 170.f);
	d = -d * (1.f/ ( 10.f));
	cv::exp(d, d);
	d = 1. + d;
	d = 1. / d;
	boostGain = 0.1 + 5. * d;


	Mat boostMap(src.size(), CV_32F);
	boostMap.setTo(0.f);
	boostGain.copyTo(boostMap,vMask);


	//filter
	llf.filter(src, destFastLLFAaptive, order * 2, sigmaMap, boostMap, level);
	gfllf.filter(src, destFourierLLFAaptive, order, sigmaMap, boostMap, level);


	cv::imwrite(".\\destFastLLF.tif", destFastLLF);
	cv::imwrite(".\\destFourierLLF.tif", destFourierLLF);
	cv::imwrite(".\\destFastLLFAaptive.tif", destFastLLFAaptive);
	cv::imwrite(".\\destFourierLLFAaptive.tif", destFourierLLFAaptive);
	return 0;
}

int main()
{
	return dbt();

	//load image
	//Mat src = imread("flower.png");
	Mat src = imread("Torace1.Dat.png", cv::IMREAD_GRAYSCALE|cv::IMREAD_ANYDEPTH);
	src.convertTo(src, CV_32F);
	src = src + 1;
	cv::log(src, src);
	cv::normalize(src, src, 0., 255., cv::NormTypes::NORM_MINMAX);
	cv::Mat vMask;
	cv::threshold(src, vMask, 0., 1., cv::ThresholdTypes::THRESH_BINARY);

	//destination image
	Mat destFastLLF, destFourierLLF, destFastLLFAaptive, destFourierLLFAaptive;

	//parameter setting
	const float sigma = 30.f;
	const float boost = 10.f;
	const int level = 10;
	const int order = 5;

	//create instance
	FastLLF llf;
	GaussianFourierLLF gfllf;

	//parameter fix filter
	llf.filter(src, destFastLLF, order * 2, sigma, boost, level);//order*2: FourierLLF requires double pyramids due to cos and sin pyramids; thus we double the order to adjust the number of pyramids.
	gfllf.filter(src, destFourierLLF, order, sigma, boost, level);

	//parameter adaptive filter
	//generate parameter maps (circle shape)
	Mat sigmaMap(src.size(), CV_32F);
	sigmaMap.setTo(sigma);
	circle(sigmaMap, Point(src.size()) / 2, src.cols / 4, Scalar::all(sigma * 2.f), cv::FILLED);

	Mat boostMap(src.size(), CV_32F);
	boostMap.setTo(boost);
	circle(boostMap, Point(src.size()) / 2, src.cols / 4, Scalar::all(boost * 2.0), cv::FILLED);

	//filter
	llf.filter(src, destFastLLFAaptive, order * 2, sigmaMap, boostMap, level);
	gfllf.filter(src, destFourierLLFAaptive, order, sigmaMap, boostMap, level);
	cv::imwrite(".\\destFastLLF.tif", destFastLLF);
	cv::imwrite(".\\destFourierLLF.tif", destFourierLLF);
	cv::imwrite(".\\destFastLLFAaptive.tif", destFastLLFAaptive);
	cv::imwrite(".\\destFourierLLFAaptive.tif", destFourierLLFAaptive);

	//imshow("src", src);
	//imshow("Fast LLF dest", destFastLLF);
	//imshow("Fourier LLF dest", destFourierLLF);
	//imshow("Fast LLF Adaptive dest", destFastLLFAaptive);
	//imshow("Fourier LLF Adaptive dest", destFourierLLFAaptive);
	//compare("LLF", destFastLLF, destFourierLLF);//quit `q` key
	//compare("LLFAdaptive", destFastLLFAaptive, destFourierLLFAaptive);//quit `q` key
	return 0;
}