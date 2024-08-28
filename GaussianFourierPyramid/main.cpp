#include "LLF.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>


#include <boost\property_tree\xml_parser.hpp>
#include <boost\program_options.hpp>
#include <boost\tokenizer.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include <filesystem>
#include <algorithm>

namespace po = boost::program_options;
namespace fs = std::filesystem;


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
void compare(const string wname, const cv::Mat& src1, const cv::Mat& src2)
{
	namedWindow(wname);
	int a = 0; createTrackbar("alpha", wname, &a, 100);
	int key = 0;
	cv::Mat show;
	double zoomLevel = 1;
	while (key != 'q')
	{
		switch (key)
		{

		case '+':
			zoomLevel += 0.1;
			break;

		case '-':
			zoomLevel -= 0.1;
			if (zoomLevel < 0.1) zoomLevel = 0.1;
			break;
		}

		cv::Mat zoomedSrc1;
		resize(src1, zoomedSrc1, Size(), zoomLevel, zoomLevel);
		cv::Mat zoomedSrc2;
		resize(src2, zoomedSrc2, Size(), zoomLevel, zoomLevel);

		addWeighted(zoomedSrc1, double(a) * 0.01, zoomedSrc2, double(100 - a) * 0.01, 0.0, show);
		imshow(wname, show);
		key = waitKey(1);
	}
}

std::vector<std::string> getFolderFiles(const std::string& folder_path)
{
	std::vector<std::string> allFiles;
	// Iterate over all files in the folder
	for (const auto& entry : fs::directory_iterator(folder_path))
	{
		// Get the file name from the path
		std::string file_name = entry.path().filename().string();
		std::string full_file_name = entry.path().string();
		allFiles.push_back(full_file_name);
	}
	return allFiles;
}


#pragma region mseEnh
static std::vector<cv::Mat> mseEnhVolume(std::vector<cv::Mat>& inputSlices, float sigmaMSE, const std::vector<float>& boostMSE, int levelMSE)
{
	// MSE filter
	MSEGaussRemap mse;

	std::vector<cv::Mat> enhSlices;
	for (const auto& slice : inputSlices)
	{
		cv::Mat filtered;

		mse.filter(slice, filtered, sigmaMSE, boostMSE, levelMSE);
		enhSlices.push_back(filtered);
	}
	return enhSlices;
}
#pragma endregion


#pragma region tests
static int testMseEnhVolume(const std::string& inputFolder, const std::string& outputFolder)
{
	//load images
	auto inputFiles = getFolderFiles(inputFolder);

	std::vector<cv::Mat> inputSlices;
	for (const auto& fn : inputFiles)
		inputSlices.push_back(imread(fn, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH));

	//parameter setting
	const float sigmaMSE = 100.f;
	const std::vector<float> boostMSE{ 1.f, 1.2f };
	const int levelMSE = 6;

	std::vector<cv::Mat> enhSlices = mseEnhVolume(inputSlices, sigmaMSE, boostMSE, levelMSE);

	if (std::filesystem::exists(outputFolder) && !std::filesystem::is_directory(outputFolder))
		return -1;


	for (auto n = 0; n < enhSlices.size(); ++n)
	{
		std::filesystem::path p{ outputFolder };
		std::filesystem::path fn{ inputFiles[n] };
		p /= fn.filename();
		p.replace_extension("tif");
		cv::imwrite(p.string(), enhSlices[n]);
	}
	return 0;
}

static int testMseEnhProjWithLog(const std::string& inputFolder, const std::string& outputFolder)
{
	//load images
	auto inputFiles = getFolderFiles(inputFolder);

	std::vector<cv::Mat> inputSlices;
	for (const auto& fn : inputFiles)
		inputSlices.push_back(imread(fn, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH));

	//parameter setting
	const float sigmaMSE = 100.f;
	const std::vector<float> boostMSE{ 1.f, 4.f,4.f,3.f,3.f,3.f,3.f,2.f,-1.f,-1.f,-1.f,-1. };
	const int levelMSE = 12;

	// MSE filter
	MSEGaussRemap mse;

	std::vector<cv::Mat> enhSlices;
	for (const auto& slice : inputSlices)
	{
		cv::Mat filtered;
		cv::add(slice, 1., filtered, cv::noArray(), CV_32F);
		cv::log(filtered, filtered);
		filtered.convertTo(filtered, CV_16U, 1000.);

		cv::Mat filteredOut;
		mse.filter(filtered, filteredOut, sigmaMSE, boostMSE, levelMSE);
		enhSlices.push_back(filteredOut);
	}

	if (std::filesystem::exists(outputFolder) && !std::filesystem::is_directory(outputFolder))
		return -1;

	for (auto n = 0; n < enhSlices.size(); ++n)
	{
		std::filesystem::path p{ outputFolder };
		std::filesystem::path fn{ inputFiles[n] };
		p /= fn.filename();
		p.replace_extension("tif");
		cv::imwrite(p.string(), enhSlices[n]);
	}
	return 0;
}

int testBilateralDenoise(std::string& inputFile, std::string& outputFile)
{
	//load image
	cv::Mat srcOriginal;
	srcOriginal = imread(inputFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

	//parameter setting
	const float sigmaRangeBilateral = 0.35f;
	const float sigmaSpaceBilateral = 1.5f;
	const int levelMSE = 2;
	const int filterSize = 15;

	// VST
	cv::Mat dn;
	srcOriginal.convertTo(dn, CV_32F);
	sqrt(dn, dn);

	MSEBilateral mse;
	cv::Mat LaplacianLevels;

	// MSE - gamma
	mse.filter(dn, LaplacianLevels, std::vector<float> {sigmaRangeBilateral}, sigmaSpaceBilateral, filterSize, levelMSE);

	// invert VST
	pow(LaplacianLevels, 2., LaplacianLevels);
	LaplacianLevels.convertTo(LaplacianLevels, CV_16U);

	cv::imwrite(outputFile, LaplacianLevels);

	return 0;
}

int testprojDenoise(const std::string& inputFolder, const std::string& inputFileFlat, const std::string& outputFolder)
{
	//load image
	cv::Mat flat;
	flat = imread(inputFileFlat, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

	//parameter setting
	const float sigmaSpace = 0.7f;
	const int levels = 4;
	const int filterSize = 5;

	projDenoise projFilter;

	//load images
	auto inputFiles = getFolderFiles(inputFolder);

	std::vector<Mat> slices;
	for (const auto& fn : inputFiles)
	{
		slices.push_back(imread(fn, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH));
	}
	const int flatBorder{ 200 };
	projFilter.apply(slices, flat, flatBorder, sigmaSpace, filterSize, levels);

	for (auto n = 0; n < slices.size(); ++n)
	{
		std::filesystem::path p{ outputFolder };
		std::filesystem::path fn{ inputFiles[n] };
		p /= fn.filename();
		p.replace_extension("png");
		cv::imwrite(p.string(), slices[n]);
	}

	return 0;
}

static int testBilateralHPF(std::string& inputFile, std::string& outputFile)
{
	//load image
	cv::Mat srcOriginal;
	srcOriginal = imread(inputFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

	//parameter setting
	const float sigmaRangeBilateral = 10.f;
	const float sigmaSpaceBilateral = 3.f;
	const int levelMSE = 4;
	const int filterSize = 15;

	cv::Mat dn;
	srcOriginal.convertTo(dn, CV_32F);

	MSEBilateral mse;
	cv::Mat LaplacianLevels;

	// MSE - gamma
	mse.filterHPF(dn, LaplacianLevels, sigmaRangeBilateral, sigmaSpaceBilateral, filterSize, levelMSE);

	cv::imwrite(outputFile, LaplacianLevels);

	return 0;
}

static int test2DSynth(std::string& inputFile, std::string& outputFile, double maskThreshold)
{
	//load image
	cv::Mat srcOriginal;
	srcOriginal = imread(inputFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

	// mask
	cv::Mat anatomicMask;
	cv::compare(srcOriginal, maskThreshold, anatomicMask, cv::CmpTypes::CMP_GT);

	//parameter setting
	const float sigmaRangeBilateral = 5.f;
	const float sigmaSpaceBilateral = 3.f;
	const int levelBilateralMSE = 1;
	const int filterSize = 15;

	cv::Mat dn;
	srcOriginal.convertTo(dn, CV_32F);

	// bilateral
	MSEBilateral bilateralFilter;
	cv::Mat bilateralEnh;
	bilateralFilter.filter(dn, bilateralEnh, std::vector<float> {sigmaRangeBilateral}, sigmaSpaceBilateral, filterSize, levelBilateralMSE);


	//parameter setting
	// MSE - gauss enh
	MSEGaussRemap mse;

	const float sigmaMSE = 100.f;
	const std::vector<float> boostMSE{ 1.f,3.f };
	const int levelMSE = 6;

	cv::Mat mseEnh;
	mse.filter(bilateralEnh, mseEnh, sigmaMSE, boostMSE, levelMSE);
	threshold(mseEnh, mseEnh, 0., 0., cv::ThresholdTypes::THRESH_TOZERO);

	cv::Mat mseEnh_16u;
	mseEnh.convertTo(mseEnh_16u, CV_16U);

	// gamma
	gammaEnh gammaFilter;
	double gamma{ 4. }, tailPercent{ 0.001 }, enlargeRangePerc{ 0.1 };

	cv::Mat gammaOut_16u;
	mseEnh_16u.copyTo(gammaOut_16u);
	std::vector<cv::Mat> inputImages{ gammaOut_16u };
	std::vector<cv::Mat> inputMasks{ anatomicMask };

	gammaFilter.applyGamma(inputImages, inputMasks, gamma, tailPercent, enlargeRangePerc);

	cv::imwrite(outputFile, gammaOut_16u);

	return 0;
}

int testMseAndGammaOrder(const std::string& inputFile, ushort maskThreshold)
{
	//parameter setting
	// MSE filter
	const float sigmaMSE = 100.f;
	const std::vector<float> boostMSE{ 1.f, 1.2f };
	const int levelMSE = 6;

	// gamma filter
	double tailPercent = 0.001;
	double enlargeRangePerc = 0.1;
	double gamma{ 3.5 };

	MSEGaussRemap mse;
	gammaEnh gammaFilter;

	//load image
	Mat src = imread(inputFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	Mat srcCopy;
	src.copyTo(srcCopy);

	// anatomic mask
	cv::Mat anatomicMask;
	cv::compare(src, maskThreshold, anatomicMask, cv::CmpTypes::CMP_GT);


	// MSE - gamma
	Mat mseFiltered;
	mse.filter(src, mseFiltered, sigmaMSE, boostMSE, levelMSE);


	Mat mseFilteredGammaCorrected;
	gammaFilter.applyGamma(mseFiltered, anatomicMask, gamma, tailPercent, enlargeRangePerc);
	//Mat mseFilteredGammaCorrectedMasked{ Mat::zeros(mseFilteredGammaCorrected.size(), mseFilteredGammaCorrected.type()) };
	//mseFilteredGammaCorrected.copyTo(mseFilteredGammaCorrectedMasked, anatomicMask);

	// gamma - MSE
	gammaFilter.applyGamma(srcCopy, anatomicMask, gamma, tailPercent, enlargeRangePerc);

	Mat gammaCorrectedMseFilter;
	mse.filter(srcCopy, gammaCorrectedMseFilter, sigmaMSE, boostMSE, levelMSE);

	threshold(gammaCorrectedMseFilter, gammaCorrectedMseFilter, 0., 0., cv::ThresholdTypes::THRESH_TOZERO);
	Mat gammaCorrectedMseFilterMasked{ Mat::zeros(gammaCorrectedMseFilter.size(), gammaCorrectedMseFilter.type()) };
	gammaCorrectedMseFilter.copyTo(gammaCorrectedMseFilterMasked, anatomicMask);

	cv::imwrite(".\\mseFilteredGammaCorrected.tif", mseFiltered);
	cv::imwrite(".\\gammaCorrectedMseFilter.tif", gammaCorrectedMseFilterMasked);

	return 0;
}

int testLLF(const std::string& inputFile, ushort maskThreshold)
{
	//load image
	Mat src = imread(inputFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

	// anatomic mask
	cv::Mat anatomicMask;

	cv::compare(src, maskThreshold, anatomicMask, cv::CmpTypes::CMP_GT);
	// normalize src (only needed for adaptive)
	cv::normalize(src, src, 0., 255., cv::NormTypes::NORM_MINMAX);

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
	Mat sigmaMap(src.size(), CV_32F);
	sigmaMap.setTo(1.f);
	sigmaMap.setTo(sigma, anatomicMask);

	cv::Mat boostGain;
	src.copyTo(boostGain);
	cv::Mat d;
	d = (boostGain - 170.f);
	d = -d * (1.f / (10.f));
	cv::exp(d, d);
	d = 1. + d;
	d = 1. / d;
	boostGain = 0.1 + 5. * d;


	Mat boostMap(src.size(), CV_32F);
	boostMap.setTo(0.f);
	boostGain.copyTo(boostMap, anatomicMask);


	//filter adaptive
	llf.filter(src, destFastLLFAaptive, order * 2, sigmaMap, boostMap, level);
	gfllf.filter(src, destFourierLLFAaptive, order, sigmaMap, boostMap, level);

	cv::imwrite(".\\destFastLLF.tif", destFastLLF);
	cv::imwrite(".\\destFourierLLF.tif", destFourierLLF);
	cv::imwrite(".\\destFastLLFAaptive.tif", destFastLLFAaptive);
	cv::imwrite(".\\destFourierLLFAaptive.tif", destFourierLLFAaptive);

	return 0;
}

int testGamma(std::string& inputFile, std::string& outputFile, ushort maskThreshold)
{
	double tailPercent = 0.001;
	double enlargeRangePerc = 0.1;
	double gamma{ 3.5 };

	gammaEnh gammaFilter;

	//load image
	Mat src = imread(inputFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

	// anatomic mask
	cv::Mat anatomicMask;
	cv::compare(src, maskThreshold, anatomicMask, cv::CmpTypes::CMP_GT);

	gammaFilter.applyGamma(src, anatomicMask, gamma, tailPercent, enlargeRangePerc);

	cv::imwrite(outputFile, src);

	return 0;
}

int testGammaVolume(const std::string& inputFolder, const std::string& outputFolder, ushort maskThreshold)
{
	double tailPercent = 0.001;
	double enlargeRangePerc = 0.1;
	double gamma{ 3.5 };

	gammaEnh gammaFilter;

	//load images
	auto inputFiles = getFolderFiles(inputFolder);

	std::vector<Mat> sliceAnatomicMask;
	std::vector<Mat> slices;
	for (const auto& fn : inputFiles)
	{
		slices.push_back(imread(fn, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH));

		// create anatomic mask
		cv::Mat anatomicMask;
		cv::compare(slices[slices.size() - 1], maskThreshold, anatomicMask, cv::CmpTypes::CMP_GT);
		sliceAnatomicMask.push_back(anatomicMask);
	}

	gammaFilter.applyGamma(slices, sliceAnatomicMask, gamma, tailPercent, enlargeRangePerc);

	for (auto n = 0; n < slices.size(); ++n)
	{
		std::filesystem::path p{ outputFolder };
		std::filesystem::path fn{ inputFiles[n] };
		p /= fn.filename();
		p.replace_extension("tif");
		cv::imwrite(p.string(), slices[n]);
	}

	return 0;
}

int testSigmoid(std::string& inputFile, std::string& outputFile, ushort maskThreshold)
{
	double tailPercent = 0.001;
	double enlargeRangePerc = 0.1;
	double gamma{ 0.15 };

	sigmoidEnh sigmoidFilter;

	//load image
	Mat src = imread(inputFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

	// anatomic mask
	cv::Mat anatomicMask;
	cv::compare(src, maskThreshold, anatomicMask, cv::CmpTypes::CMP_GT);

	sigmoidFilter.apply(src, anatomicMask, gamma, tailPercent, enlargeRangePerc);

	cv::imwrite(outputFile, src);

	return 0;
}

int testSigmoidVolume(const std::string& inputFolder, const std::string& outputFolder, ushort maskThreshold)
{
	double tailPercent = 0.001;
	double enlargeRangePerc = 0.1;
	double sigma{ 0.15 };

	sigmoidEnh sigmoidFilter;

	//load images
	auto inputFiles = getFolderFiles(inputFolder);

	std::vector<Mat> sliceAnatomicMask;
	std::vector<Mat> slices;
	for (const auto& fn : inputFiles)
	{
		slices.push_back(imread(fn, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH));

		// create anatomic mask
		cv::Mat anatomicMask;
		cv::compare(slices[slices.size() - 1], maskThreshold, anatomicMask, cv::CmpTypes::CMP_GT);
		sliceAnatomicMask.push_back(anatomicMask);
	}

	sigmoidFilter.apply(slices, sliceAnatomicMask, sigma, tailPercent, enlargeRangePerc);

	for (auto n = 0; n < slices.size(); ++n)
	{
		std::filesystem::path p{ outputFolder };
		std::filesystem::path fn{ inputFiles[n] };
		p /= fn.filename();
		p.replace_extension("tif");
		cv::imwrite(p.string(), slices[n]);
	}

	return 0;
}

int testMSEBlend(std::string& inputFile1, std::string& inputFile2, std::string& inputFileMask, std::string& outputFile)
{
	MSEBlend blendFilter;

	//load image
	Mat src1 = imread(inputFile1, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	Mat src2 = imread(inputFile2, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	Mat mask = imread(inputFileMask, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	Mat result;

	const int levels = 4;
	blendFilter.filter(src1, src2, mask, result, levels);

	cv::imwrite(outputFile, result);

	return 0;
}

int testLR(std::string& inputFile, std::string& outputFile)
{
	MSEGaussRemap filter;

	//load image
	Mat src = imread(inputFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	Mat result;
	src.convertTo(src, CV_32F);

	const int levels = 9;
	const float sigmaSpace{ 100 };
	filter.filter(src, result, sigmaSpace, std::vector<float>{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, -0.8f, -0.9f, -1.f}, levels);

	cv::imwrite(outputFile, result);

	return 0;
}

#pragma endregion




// Function to calculate the arc length of a polyline
double arcLength(const vector<Point>& polyline) {
	double length = 0.0;
	for (size_t i = 1; i < polyline.size(); i++) {
		length += norm(polyline[i] - polyline[i - 1]);
	}
	return length;
}

// Function to sample points along a polyline with equal spacing
double samplePoints(const vector<Point>& polyline, vector<Point2f>& points, int numPoints) {
	double totalLength = arcLength(polyline);
	double segmentLength = totalLength / (numPoints - 1);

	points.push_back(Point2f(polyline[0]));
	double accumulatedLength = 0.0;
	for (size_t i = 1; i < polyline.size(); i++) {
		double segment = norm(polyline[i] - polyline[i - 1]);
		double residual{ segment };
		while (residual + accumulatedLength >= segmentLength)
		{
			residual -= (segmentLength - accumulatedLength);
			accumulatedLength = 0.0;

			double ratio = (segment - residual) / segment;
			Point2f newPoint = polyline[i - 1] + ratio * (polyline[i] - polyline[i - 1]);
			points.push_back(newPoint);
		}
		accumulatedLength += residual;
	}
	if (points.size() < numPoints) {
		points.push_back(Point2f(polyline.back()));
	}
	return segmentLength;
}
template <class T>
T bilinearInterpolate(const Mat& image, Point2f p)
{
	float x{ p.x }, y{ p.y };
	int x1 = static_cast<int>(x);
	int y1 = static_cast<int>(y);
	int x2 = x1 + 1;
	int y2 = y1 + 1;

	if (x1 < 0 || x2 >= image.cols || y1 < 0 || y2 >= image.rows)
		return 0; // Return zero for out-of-bounds

	float a = x - x1;
	float b = y - y1;

	// Get pixel values at the four corners
	T I11 = image.at<T>(y1, x1);
	T I12 = image.at<T>(y2, x1);
	T I21 = image.at<T>(y1, x2);
	T I22 = image.at<T>(y2, x2);

	// Perform bilinear interpolation
	T I = (1 - a) * (1 - b) * I11 +
		a * (1 - b) * I21 +
		(1 - a) * b * I12 +
		a * b * I22;

	return I;
}



vector<vector<Point2f>> contractPolyline(const vector<Point2f>& polyline, vector<Point2f>& contractedPolyline, Mat gradX, Mat gradY, Mat distance, float thresholdDist)
{
	// Contract the polyline by moving points along the gradient direction
	vector<vector<Point2f>> pathTrace;
	float contractionStep = 4.0f;

	contractedPolyline.clear();

	for (const auto& point : polyline)
	{
		vector<Point2f> currentPath;
		bool borderCollision{ false };

		Point2f newPoint{ point };
		currentPath.push_back(newPoint);
		// distance.at<float>(cvRound(newPoint.y), cvRound(newPoint.x)
		while ((bilinearInterpolate<float>(distance, newPoint) < thresholdDist) && !borderCollision)
		{
			float dx = (float)bilinearInterpolate<double>(gradX, newPoint);
			float dy = (float)bilinearInterpolate<double>(gradY, newPoint);

			Point2f t = Point2f((newPoint.x + dx * contractionStep), (newPoint.y + dy * contractionStep));

			int x = cvRound(t.x);
			int y = cvRound(t.y);
			if (x - 1 >= 0 && x + 1 < gradX.cols && y - 1 >= 0 && y + 1 < gradX.rows)
			{
				if (newPoint == t)
				{
					// null gradient use previous point ********* TODO
					borderCollision = true;
				}
				else
				{
					newPoint = t;
					currentPath.push_back(newPoint);
				}
			}
			else
				borderCollision = true;
		}
		pathTrace.push_back(currentPath);
		contractedPolyline.push_back(newPoint);
	}

	// verify degenerative condition
	double meanDistance{ 0. };
	for (size_t p = 0; p < contractedPolyline.size() - 1; ++p)
		meanDistance += norm(contractedPolyline[p] - contractedPolyline[p + 1]);
	meanDistance /= double(contractedPolyline.size() - 1);

	for (size_t p = 0; p < contractedPolyline.size() - 1; ++p)
	{
		double pDist = norm(contractedPolyline[p] - contractedPolyline[p + 1]);
		if (pDist < meanDistance * 0.1)
		{
			if (p != 0)
			{
				// move p toward prev point
				double ratio = 0.1;
				Point2f newPoint = contractedPolyline[p] + ratio * (contractedPolyline[p - 1] - contractedPolyline[p]);
				contractedPolyline[p] = newPoint;
			}

			if (p + 1 < (contractedPolyline.size() - 1))
			{
				// move p+1 toward next point
				double ratio = 0.1;
				Point2f newPoint = contractedPolyline[p + 1] + ratio * (contractedPolyline[p + 2] - contractedPolyline[p + 1]);
				contractedPolyline[p] = newPoint;
			}
		}
	}
	return pathTrace;
}

vector<Point> findPolyLineExt(const cv::Mat mask)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// If no contours are found, exit
	if (contours.empty()) {
		cerr << "No contours found in the skeleton" << endl;
		return vector<Point>();
	}

	// Step 3: Extract the longest contour as the polyline
	size_t longestContourIndex = 0;
	double maxLength = 0.0;
	for (size_t i = 0; i < contours.size(); i++) {
		double length = arcLength(contours[i], false);
		if (length > maxLength) {
			maxLength = length;
			longestContourIndex = i;
		}
	}

	vector<Point> polyline = contours[longestContourIndex];

	// remove closing point TODO
	polyline.pop_back();

	return polyline;
}


void skeletonize(const Mat& src, Mat& dst) {
	dst = src.clone();
	dst /= 255; // Normalize to binary
	Mat skel(dst.size(), CV_8UC1, Scalar(0));
	Mat temp;
	Mat eroded;
	Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

	bool done;
	do {
		erode(dst, eroded, element);
		dilate(eroded, temp, element); // temp = open(dst)
		subtract(dst, temp, temp);
		bitwise_or(skel, temp, skel);
		eroded.copyTo(dst);

		done = (countNonZero(dst) == 0);
	} while (!done);

	skel *= 255; // Convert back to original scale
	skel.copyTo(dst);
}

void drawGrid(Mat& image, int gridSize)
{

	// Draw vertical lines
	for (int x = gridSize; x < image.cols; x += gridSize) {
		Scalar gridColor = Scalar((x / gridSize) % 255); // Color of the grid (white for grayscale)
		line(image, Point(x, 0), Point(x, image.rows), gridColor);
	}

	// Draw horizontal lines
	for (int y = gridSize; y < image.rows; y += gridSize) {
		Scalar gridColor = Scalar((y / gridSize) % 255); // Color of the grid (white for grayscale)
		line(image, Point(0, y), Point(image.cols, y), gridColor);
	}
}

Mat points2Mat(const vector<Point2f>& points)
{
	// Create a Mat with n rows and 2 columns
	Mat mat((int)points.size(), 2, CV_32F);

	// Copy data from vector to Mat
	for (int i = 0; i < points.size(); ++i)
	{
		mat.at<float>(i, 0) = points[i].x;
		mat.at<float>(i, 1) = points[i].y;
	}
	return mat;
}

//#define BAND_OVERLAY_GRID
#define BAND_DISPLAY_RESULT

// Function to warp the curved band to a rectangle
Mat warpBandToRectangle(Mat& image, const Mat& anatomicMask, int distThreshold, double scaleF)
{

	cv::Mat cvDistance;
	cv::distanceTransform(anatomicMask, cvDistance, cv::DIST_L2, cv::DIST_MASK_PRECISE, CV_32F);

	// Define the two polylines
	cv::Mat bandExt;
	bandExt = (cvDistance > 0.);
	vector<Point> polyline1 = findPolyLineExt(bandExt);

	// Compute the gradient of the distance transform
	Mat gradX, gradY;
	Sobel(cvDistance, gradX, CV_64F, 1, 0, 3);
	Sobel(cvDistance, gradY, CV_64F, 0, 1, 3);

	// Normalize the gradients
	Mat mag;
	magnitude(gradX, gradY, mag);
	mag = mag + std::numeric_limits<double>::epsilon();
	gradX /= mag;
	gradY /= mag;


	int numPoints = 32;
	vector<Point2f> points1, points2;
	vector<Point2f> points0;

	// Sample points along the polylines
	double pointsDistance = samplePoints(polyline1, points0, numPoints);

	// remove first and last
	points1.assign(points0.begin() + 1, points0.end() - 1);

	// Define the target rectangle dimensions similar to sampled zone
	int rectWidth = int(double(distThreshold) / scaleF);
	int rectHeight = int(double(points1.size()) * pointsDistance);

	auto pathTrace = contractPolyline(points1, points2, gradX, gradY, cvDistance, (float)(distThreshold / scaleF));

#if defined(BAND_DISPLAY_RESULT)
	/////////////////////////////////// display
	Mat backtorgb = image.clone();
	normalize(image, backtorgb, 0., 255., cv::NormTypes::NORM_MINMAX);
	backtorgb.convertTo(backtorgb, CV_8U);
	Mat display1;
	cvtColor(backtorgb, backtorgb, ColorConversionCodes::COLOR_GRAY2RGB);
	backtorgb.copyTo(display1);
	imshow("source", backtorgb);

	vector<Point> v1;
	for (const auto& v : points1)
	{
		cv::Point p((int)v.x, (int)v.y);
		drawMarker(backtorgb, p, Scalar(0, 0, 255), MarkerTypes::MARKER_CROSS);
		v1.push_back(p);
	}
	polylines(backtorgb, v1, false, Scalar(0, 0, 255), 1);

	vector<Point> v2;
	for (const auto& v : points2)
	{
		cv::Point p((int)v.x, (int)v.y);
		drawMarker(backtorgb, p, Scalar(0, 0, 255));
		v2.push_back(p);
	}
	polylines(backtorgb, v2, false, Scalar(0, 0, 255), 1);

	// gradient ascent path
	for (const auto& p : pathTrace)
	{
		vector<Point> iPoint(p.size());
		std::transform(p.begin(), p.end(), iPoint.begin(), [](const auto& pf)
			{
				return Point((int)pf.x, (int)pf.y);
			});
		polylines(backtorgb, iPoint, false, Scalar(0, 0, 255), 1);
	}

	// Display the result
	drawGrid(backtorgb, 10);
	imshow("polylines", backtorgb);
	waitKey();
	/////////////////////////////////// display
#endif

	// Output image
	Mat rectImage = Mat::zeros(rectHeight, rectWidth, image.type());

	// Warp each segment to the corresponding part of the rectangle
	float step = float(rectHeight) / float(points1.size() - 1);

	// fixed destination rectangle
	vector<Point2f> dstQuad = {
		Point2f(0.f, 0.f),
		Point2f(0.f, step),
		Point2f(float(rectWidth), step),
		Point2f(float(rectWidth), 0.f)
	};

#if defined(BAND_OVERLAY_GRID)
	/////////////////////////////////// display
	// drawGrid(image, 10);
	for (const auto& p : pathTrace)
	{
		vector<Point> iPoint(p.size());
		std::transform(p.begin(), p.end(), iPoint.begin(), [](const auto& pf)
			{
				return Point((int)pf.x, (int)pf.y);
			});
		polylines(image, iPoint, false, Scalar(0, 0, 0), 2);
	}

	for (int i = 0; i < points1.size() - 1; i++) {
		vector<Point2f> srcQuad = { points1[i], points1[i + 1], points2[i + 1], points2[i] };
		circle(image, srcQuad[0], 5, Scalar(0, 0, 0));
		circle(image, srcQuad[1], 5, Scalar(0, 0, 0));
		circle(image, srcQuad[2], 5, Scalar(0, 0, 0));
		circle(image, srcQuad[3], 5, Scalar(0, 0, 0));
	}
	/////////////////////////////////// display
#endif

	for (int i = 0; i < points1.size() - 1; i++) {
		vector<Point2f> srcQuad = { points1[i], points1[i + 1], points2[i + 1], points2[i] };
		cv::Mat r = rectImage.rowRange(int(float(i) * step), std::min(int(float((i + 1) * step)), rectImage.rows));

		Mat output;
		display1.copyTo(output);
		drawMarker(output, srcQuad[0], Scalar(0, 0, 255));
		drawMarker(output, srcQuad[1], Scalar(0, 0, 255));
		drawMarker(output, srcQuad[2], Scalar(0, 0, 255));
		drawMarker(output, srcQuad[3], Scalar(0, 0, 255));

		imshow("markers", output);
		waitKey(0);
		destroyWindow("markers");
		Mat perspectiveMatrix = getPerspectiveTransform(srcQuad, dstQuad);

		Mat homografyMatrix = findHomography(srcQuad, dstQuad);

		warpPerspective(image, r, perspectiveMatrix, r.size(), INTER_LINEAR, BORDER_TRANSPARENT);

		vector<Point2f> dst;
		perspectiveTransform(srcQuad, dst, perspectiveMatrix);
		cout << perspectiveMatrix << endl;
		cout << srcQuad << std::endl;
		cout << dst << endl;
	}

	Mat display;
	rectImage.copyTo(display);
	normalize(display, display, 0., 255, NormTypes::NORM_MINMAX);
	display.convertTo(display, CV_8U);
	imshow("rect", display);
	waitKey(0);

	return rectImage;
}

// Function to warp the curved band to a rectangle
Mat warpBandToRectangleOld(const Mat& image, const Mat& anatomicMask, int distThreshold, double scaleF)
{
	// Define the target rectangle dimensions
	int rectWidth = 1000;
	int rectHeight = distThreshold;

	cv::Mat cvDistance;
	cv::distanceTransform(anatomicMask, cvDistance, cv::DIST_L2, cv::DIST_MASK_PRECISE, CV_32F);

	// Define the two polylines
	cv::Mat bandExt;
	bandExt = (cvDistance > 0.);
	vector<Point> polyline1 = findPolyLineExt(bandExt);

	bandExt = (cvDistance > (double)distThreshold / scaleF);
	vector<Point> polyline2 = findPolyLineExt(bandExt);

	// Compute the gradient of the distance transform
	Mat gradX, gradY;
	Sobel(cvDistance, gradX, CV_64F, 1, 0, 3);
	Sobel(cvDistance, gradY, CV_64F, 0, 1, 3);

	// Normalize the gradients
	Mat mag;
	magnitude(gradX, gradY, mag);
	mag = mag + std::numeric_limits<double>::epsilon();
	gradX /= mag;
	gradY /= mag;


	int numPoints = 20;
	vector<Point2f> points1, points2;

	double epsilon = 2.0; // Adjust epsilon for desired approximation accuracy
	vector<Point> simplifiedPolyline1;
	approxPolyDP(polyline1, simplifiedPolyline1, epsilon, false);

	vector<Point> simplifiedPolyline2;
	approxPolyDP(polyline2, simplifiedPolyline2, epsilon, false);

	// Sample points along the polylines
	samplePoints(polyline1, points1, numPoints);
	samplePoints(polyline2, points2, numPoints);

	contractPolyline(points1, points2, gradX, gradY, cvDistance, (float)(distThreshold / scaleF));

	Mat outputImage = image.clone();
	vector<Point> v1;
	for (const auto& v : points1)
	{
		cv::Point p((int)v.x, (int)v.y);
		drawMarker(outputImage, p, Scalar(0, 0, 255));
		v1.push_back(p);
	}
	polylines(outputImage, v1, false, Scalar(0, 0, 255), 1);

	// Draw the simplified polyline
	vector<Point> v2;
	for (const auto& v : points2)
	{
		cv::Point p((int)v.x, (int)v.y);
		drawMarker(outputImage, p, Scalar(0, 0, 255));
		v2.push_back(p);
	}
	polylines(outputImage, v2, false, Scalar(0, 0, 255), 1);

	// Display the result
	imshow("Simplified Polyline", outputImage);
	waitKey(0);
	destroyWindow("Simplified Polyline");

	// Output image
	Mat rectImage = Mat::zeros(rectHeight, rectWidth, image.type());

	// Warp each segment to the corresponding part of the rectangle
	vector<Point2f> dstQuad = {
			Point2f(0.f, 0.f),
			Point2f(float(rectWidth) / float(numPoints - 1), 0.f),
			Point2f(float(rectWidth) / float(numPoints - 1), float(rectHeight)),
			Point2f(0.f, float(rectHeight))
	};


	float step = float(rectWidth) / float(numPoints - 1);
	for (int i = 0; i < numPoints - 1; i++) {
		vector<Point2f> srcQuad = { points1[i], points1[i + 1], points2[i + 1], points2[i] };
		cv::Mat r = rectImage.colRange(int(float(i) * step), std::min(int(float((i + 1) * step)), rectImage.cols));

		Mat output;
		image.copyTo(output);
		output.convertTo(output, CV_8U);
		drawMarker(output, srcQuad[0], Scalar(0, 0, 255));
		drawMarker(output, srcQuad[1], Scalar(0, 0, 255));
		drawMarker(output, srcQuad[2], Scalar(0, 0, 255));
		drawMarker(output, srcQuad[3], Scalar(0, 0, 255));

		imshow("Simplified Polyline", output);
		waitKey(0);

		destroyWindow("Simplified Polyline");
		Mat perspectiveMatrix = getPerspectiveTransform(srcQuad, dstQuad);
		warpPerspective(image, r, perspectiveMatrix, r.size(), INTER_LINEAR, BORDER_TRANSPARENT);
	}

	return rectImage;
}



int testBand(std::string inputFile, std::string inputFlatFile, std::string outputFile, ushort maskThreshold, int distThreshold, double scaleF = 4.)
{
	//load image
	Mat src = imread(inputFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	Mat flat = imread(inputFlatFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

	src.convertTo(src, CV_32F);
	cv::log(src, src);

	flat.convertTo(flat, CV_32F);
	cv::log(flat, flat);

	src = flat - src;
	src = src * 1000.;

	cv::resize(src, src, cv::Size(), 1. / scaleF, 1. / scaleF);
	Mat srcCropped;
	srcCropped = src.colRange(Range(50, src.cols - 1));
	// anatomic mask
	cv::Mat anatomicMask;
	cv::compare(srcCropped, maskThreshold, anatomicMask, cv::CmpTypes::CMP_GT);

	// Debug 

	Mat filtered;
#if defined(BAND_USE_MSE)
	// MSE filter
	MSEGaussRemap mse;
	Mat filtered;
	const float sigmaMSE = 100.f;
	const std::vector<float> boostMSE{ 1.f,3.f,3.f,3.f,3.f,3.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f };
	const int levelMSE = 10;
	mse.filter(srcCropped, filtered, sigmaMSE, boostMSE, levelMSE);
#else
	srcCropped.copyTo(filtered);
#endif

	// Warp the band to a rectangle
	Mat rectImage = warpBandToRectangle(filtered, anatomicMask, distThreshold, scaleF);

	std::filesystem::path p{ outputFile };
	std::filesystem::path fn{ inputFile };
	p.replace_filename(fn.filename());
	p.replace_extension(".tif");
	cv::imwrite(p.string(), src);
	imwrite(outputFile, rectImage);
	return 0;
}

static std::vector<cv::Mat> maskedFilterGauss(cv::Mat& cvProj, cv::Mat& cvSmaskData, double sigma)
{
	cv::Mat binaryMask;
	cv::threshold(cvSmaskData, binaryMask, 0., 1, cv::THRESH_BINARY);
	cv::Mat gaussMask;
	cv::GaussianBlur(binaryMask, gaussMask, cv::Size(), sigma);

	cv::Mat projFiltered;
	cv::multiply(cvProj, binaryMask, projFiltered);
	cv::GaussianBlur(projFiltered, projFiltered, cv::Size(), sigma);

	gaussMask += std::numeric_limits<float>::epsilon();
	cv::divide(projFiltered, gaussMask, projFiltered);

	cv::Mat result;
	cv::multiply(projFiltered, binaryMask, result);

	std::vector<cv::Mat> r;
	r.push_back(result);
	r.push_back(binaryMask);
	return r;
}

static int testEqualize(std::string& inputFile, std::string& outputFile, double maskThreshold)
{
	//load image
	cv::Mat srcOriginal;
	srcOriginal = imread(inputFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

	// mask
	cv::Mat anatomicMask;
	cv::compare(srcOriginal, maskThreshold, anatomicMask, cv::CmpTypes::CMP_GT);
	anatomicMask.convertTo(anatomicMask, CV_32F);
	cv::threshold(srcOriginal, anatomicMask, maskThreshold, 1, cv::THRESH_BINARY);

	const double sigma{ 40 };
	std::vector<cv::Mat> filteredGauss = maskedFilterGauss(srcOriginal, anatomicMask, sigma);
	cv::Mat srcFiltered{ filteredGauss[0] };
	cv::Mat binaryMask{ filteredGauss[1] };
	binaryMask = (binaryMask > 0);

	cv::Mat equalizedResult;



	cv::Mat cvDistance;
	cv::Mat mask8u;
	anatomicMask.copyTo(mask8u);
	mask8u = (mask8u > 0);
	cv::distanceTransform(mask8u, cvDistance, cv::DIST_L2, cv::DIST_MASK_PRECISE, CV_32F);

	cv::Mat compressedMask;
	compressedMask = (cvDistance > 200);


	// Compute the mean and standard deviation only for the pixels selected by the mask
	cv::Scalar meanValue, stdDevValue;
	cv::meanStdDev(srcFiltered, meanValue, stdDevValue, compressedMask);

	//std::vector<float> thresholds{ 0.875f,0.95f,1.f,1.1f,1.2f };
	std::vector<float> thresholds;
	const float rangeMin{ 0.8f };
	const float rangeMax{ 1.2f };
	const float rangeStep{ 11.f };
	const float delta = (rangeMax - rangeMin) / rangeStep;
	for (auto n = 0; n < rangeStep; ++n)
		thresholds.push_back(rangeMin + float(n) * delta);

	std::vector <cv::Mat> partialRes;
	cv::Mat accum = cv::Mat::zeros(srcFiltered.size(), CV_32F);
	for (int n = 0; n < thresholds.size(); ++n)
	{
		cv::Mat partial = cv::Mat::ones(srcFiltered.size(), srcFiltered.type());

		const float threshold = thresholds.at(n) * float(meanValue[0]);

		// Extract the points from the mask
		std::vector<cv::Point3f> points;
		for (int y = 0; y < srcFiltered.rows; ++y)
		{
			for (int x = 0; x < srcFiltered.cols; ++x)
			{
				if (binaryMask.at<uchar>(y, x) > 0)
				{
					const float z = srcFiltered.at<float>(y, x);
					if (z <= threshold)
						partial.at<float>(y, x) = z / threshold;
				}
			}
		}
		partialRes.push_back(partial);

		accum += partial;
	}
	accum /= cv::Scalar(double(thresholds.size()));

	cv::Mat equalizer;
	const float powerFactor{ 0.8f };
	cv::pow(accum, powerFactor, equalizer);

	cv::divide(srcOriginal, equalizer, equalizedResult);
	cv::Mat equalizedFiltered;
	cv::divide(srcFiltered, equalizer, equalizedFiltered);
	cv::multiply(equalizedResult, anatomicMask, equalizedResult);
	cv::multiply(equalizer, anatomicMask, equalizer);
	cv::multiply(accum, anatomicMask, accum);

	equalizedResult.copyTo(srcFiltered);
	cv::imwrite(outputFile, equalizedResult);


	return 0;
}


void enhSliceConspicuity(const cv::Mat& image, cv::Mat& enhImage, cv::Mat& weight, double sigma_px = 2.0, int kernelSize = 13, double power_factor = 1.2)
{
	cv::Mat grad_x, grad_y;
	cv::Sobel(image, grad_x, CV_32F, 1, 0, 3, 0.001); // Gradient in x direction
	cv::Sobel(image, grad_y, CV_32F, 0, 1, 3, 0.001); // Gradient in y direction

	cv::Mat gradient_magnitude;
	cv::magnitude(grad_x, grad_y, gradient_magnitude);

	cv::Mat smoothed_gradient;
	cv::GaussianBlur(gradient_magnitude, smoothed_gradient, cv::Size(kernelSize, kernelSize), sigma_px);

	cv::pow(smoothed_gradient, power_factor, weight);
	cv::Mat image_32f;
	image.convertTo(image_32f, CV_32F);
	cv::multiply(image_32f, weight, enhImage);
}



static int testVolumeConspicuity(const std::string& inputFolder, const std::string& outputFolder)
{
	//load images
	auto inputFiles = getFolderFiles(inputFolder);

	std::vector<cv::Mat> inputSlices;
	for (const auto& fn : inputFiles)
		inputSlices.push_back(imread(fn, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH));

	//parameter setting

	std::vector<cv::Mat> enhSlices(inputSlices.size());
	std::vector<cv::Mat> weights(inputSlices.size());
	for (auto n = 0; n < inputSlices.size(); ++n)
		enhSliceConspicuity(inputSlices.at(n), enhSlices.at(n), weights.at(n));

	if (std::filesystem::exists(outputFolder) && !std::filesystem::is_directory(outputFolder))
		return -1;

	for (auto n = 0; n < enhSlices.size(); ++n)
	{
		std::filesystem::path p{ outputFolder };
		std::filesystem::path fn{ inputFiles[n] };
		p /= fn.filename();
		p.replace_extension("tif");

		std::filesystem::path w{ p };
		std::ostringstream ost;
		ost << "w_" << std::setfill('0') << setw(3) << n;
		w.replace_filename(ost.str());
		w.replace_extension("tif");

		cv::imwrite(p.string(), enhSlices[n]);
		cv::imwrite(w.string(), weights[n]);
	}
	return 0;
}


int testAdaptiveGammaVolume(const std::string& inputFolder, const std::string& outputFolder, ushort maskThreshold)
{
	double gamma{ 1.3 };

	gammaEnh gammaFilter;

	//load images
	auto inputFiles = getFolderFiles(inputFolder);

	std::vector<Mat> sliceAnatomicMask;
	std::vector<Mat> slices;
	for (const auto& fn : inputFiles)
	{
		slices.push_back(imread(fn, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH));

		// create anatomic mask
		cv::Mat anatomicMask;
		cv::compare(slices[slices.size() - 1], maskThreshold, anatomicMask, cv::CmpTypes::CMP_GT);
		sliceAnatomicMask.push_back(anatomicMask);
	}

	gammaFilter.applyAdaptiveGamma(slices, sliceAnatomicMask);
	//gammaFilter.applyAdaptiveGammaCDF (slices, sliceAnatomicMask);

	for (auto n = 0; n < slices.size(); ++n)
	{
		std::filesystem::path p{ outputFolder };
		std::filesystem::path fn{ inputFiles[n] };
		p /= fn.filename();
		p.replace_extension("tif");
		cv::imwrite(p.string(), slices[n]);
	}

	return 0;
}


int testDoubleBilateral(std::string& inputFile, std::string& outputFile)
{
	//load image

	cv::Mat centralProjPos;
	cv::Mat centralAvgProj;
	cv::Mat centralSynth2D;
	cv::Mat centralProjMsk;

	centralProjPos = cv::imread("d:\\tmp\\synth2D\\projPos.tif", cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	centralAvgProj = cv::imread("d:\\tmp\\synth2D\\projAvg.tif", cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	centralSynth2D = cv::imread("d:\\tmp\\synth2D\\proj2D.tif", cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	centralProjMsk = cv::imread("d:\\tmp\\synth2D\\projMask.tif", cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

	cv::Mat erodedMask;
	cv::erode(centralProjMsk, erodedMask, cv::Mat::ones(15, 15, CV_32F));


	MSEWeightedBlend wFilter;
	cv::Mat synth2D;
	int levels = 7;
	wFilter.filter(centralAvgProj, centralProjPos, erodedMask, synth2D, levels);

	cv::imwrite("d:\\tmp\\synth2d\\enh.tif", synth2D);


	MSEBilateral bl;

	cv::Mat inside = centralProjMsk.clone();
	inside = inside > 0;
	cv::Mat fullOutside = inside == 0;

	cv::Mat et;
	cv::distanceTransform(fullOutside, et,
		cv::DIST_L2,
		cv::DIST_MASK_PRECISE,
		CV_32F);

	cv::Mat etb;
	cv::threshold(et, etb, 128, 1, cv::THRESH_BINARY_INV);
	etb = (etb > 0);
	cv::bitwise_xor(etb, fullOutside, etb);
	cv::bitwise_not(etb, etb);

	cv::Mat outside;
	cv::bitwise_or(etb, inside, outside);
	cv::bitwise_not(outside, outside);

	auto mean = cv::mean(centralAvgProj, inside)[0];
	cv::Mat build = centralAvgProj.clone();
	build.setTo(mean, outside);


	cv::Mat inpainted;
	double inpaintRadius{ double(128) };
	// INPAINT_TELEA better than INPAINT_NS
	cv::inpaint(build, etb, inpainted, inpaintRadius, cv::INPAINT_TELEA);


	cv::Mat destHF;
	double sigma_range = 5.;
	double sigma_space = 1.;
	int filterSize = 7;
	int levelsBL = 6;

	bl.filterHPF(build, destHF, sigma_range, sigma_space, filterSize, levelsBL);
	return 0;
}





#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>

// Haar Wavelet Coefficients
const float haarLowPassFilter[] = { 1.0f / std::sqrt(2.0f), 1.0f / std::sqrt(2.0f) };
const float haarHighPassFilter[] = { 1.0f / std::sqrt(2.0f), -1.0f / std::sqrt(2.0f) };

// Daubechies 4 Wavelet Coefficients
const float h0 = (1 + std::sqrt(3)) / (4 * std::sqrt(2));
const float h1 = (3 + std::sqrt(3)) / (4 * std::sqrt(2));
const float h2 = (3 - std::sqrt(3)) / (4 * std::sqrt(2));
const float h3 = (1 - std::sqrt(3)) / (4 * std::sqrt(2));
const float g0 = h3;
const float g1 = -h2;
const float g2 = h1;
const float g3 = -h0;

// Shannon–Cosine Wavelet Coefficients
const float shannonCosineLowPassFilter[] = { 0.5f, 0.5f };
const float shannonCosineHighPassFilter[] = { 0.5f, -0.5f };

// Forward Haar Transform
void dwtHaar(const cv::Mat& src, cv::Mat& cA, cv::Mat& cH, cv::Mat& cV, cv::Mat& cD) {
	cv::Mat src_f;
	src.convertTo(src_f, CV_32F);

	int width = src.cols / 2;
	int height = src.rows / 2;

	cA = cv::Mat::zeros(height, width, CV_32F);
	cH = cv::Mat::zeros(height, width, CV_32F);
	cV = cv::Mat::zeros(height, width, CV_32F);
	cD = cv::Mat::zeros(height, width, CV_32F);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float a = 0.0f, h = 0.0f, v = 0.0f, d = 0.0f;

			for (int ky = 0; ky < 2; ++ky) {
				for (int kx = 0; kx < 2; ++kx) {
					float pixel = src_f.at<float>(y * 2 + ky, x * 2 + kx);

					a += pixel * haarLowPassFilter[ky] * haarLowPassFilter[kx];
					h += pixel * haarLowPassFilter[ky] * haarHighPassFilter[kx];
					v += pixel * haarHighPassFilter[ky] * haarLowPassFilter[kx];
					d += pixel * haarHighPassFilter[ky] * haarHighPassFilter[kx];
				}
			}

			cA.at<float>(y, x) = a;
			cH.at<float>(y, x) = h;
			cV.at<float>(y, x) = v;
			cD.at<float>(y, x) = d;
		}
	}
}

// Forward Daubechies 4 Transform
void dwtDaubechies4(const cv::Mat& src, cv::Mat& cA, cv::Mat& cH, cv::Mat& cV, cv::Mat& cD) {
	cv::Mat src_f;
	src.convertTo(src_f, CV_32F);

	int width = src.cols / 2;
	int height = src.rows / 2;

	cA = cv::Mat::zeros(height, width, CV_32F);
	cH = cv::Mat::zeros(height, width, CV_32F);
	cV = cv::Mat::zeros(height, width, CV_32F);
	cD = cv::Mat::zeros(height, width, CV_32F);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float a = 0.0f, h = 0.0f, v = 0.0f, d = 0.0f;

			for (int ky = 0; ky < 2; ++ky) {
				for (int kx = 0; kx < 2; ++kx) {
					float pixel = src_f.at<float>(y * 2 + ky, x * 2 + kx);

					a += pixel * ((ky == 0 ? h0 : h2) * (kx == 0 ? h0 : h2));
					h += pixel * ((ky == 0 ? h0 : h2) * (kx == 0 ? g0 : g2));
					v += pixel * ((ky == 0 ? g0 : g2) * (kx == 0 ? h0 : h2));
					d += pixel * ((ky == 0 ? g0 : g2) * (kx == 0 ? g0 : g2));
				}
			}

			cA.at<float>(y, x) = a;
			cH.at<float>(y, x) = h;
			cV.at<float>(y, x) = v;
			cD.at<float>(y, x) = d;
		}
	}
}

// Forward Shannon–Cosine Transform
void dwtShannonCosine(const cv::Mat& src, cv::Mat& cA, cv::Mat& cH, cv::Mat& cV, cv::Mat& cD) {
	cv::Mat src_f;
	src.convertTo(src_f, CV_32F);

	int width = src.cols / 2;
	int height = src.rows / 2;

	cA = cv::Mat::zeros(height, width, CV_32F);
	cH = cv::Mat::zeros(height, width, CV_32F);
	cV = cv::Mat::zeros(height, width, CV_32F);
	cD = cv::Mat::zeros(height, width, CV_32F);

	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			float a = 0.0f, h = 0.0f, v = 0.0f, d = 0.0f;

			for (int ky = 0; ky < 2; ++ky) {
				for (int kx = 0; kx < 2; ++kx) {
					float pixel = src_f.at<float>(y * 2 + ky, x * 2 + kx);

					a += pixel * shannonCosineLowPassFilter[ky] * shannonCosineLowPassFilter[kx];
					h += pixel * shannonCosineLowPassFilter[ky] * shannonCosineHighPassFilter[kx];
					v += pixel * shannonCosineHighPassFilter[ky] * shannonCosineLowPassFilter[kx];
					d += pixel * shannonCosineHighPassFilter[ky] * shannonCosineHighPassFilter[kx];
				}
			}

			cA.at<float>(y, x) = a;
			cH.at<float>(y, x) = h;
			cV.at<float>(y, x) = v;
			cD.at<float>(y, x) = d;
		}
	}
}

// Multilevel Wavelet Transform
void multilevelDWT(const cv::Mat& src, std::vector<cv::Mat>& cA, std::vector<cv::Mat>& cH, std::vector<cv::Mat>& cV, std::vector<cv::Mat>& cD, int levels, const std::string& waveletType) {
	cv::Mat currentSrc = src;
	for (int i = 0; i < levels; ++i) {
		cv::Mat tempA, tempH, tempV, tempD;
		if (waveletType == "haar") {
			dwtHaar(currentSrc, tempA, tempH, tempV, tempD);
		}
		else if (waveletType == "daubechies4") {
			dwtDaubechies4(currentSrc, tempA, tempH, tempV, tempD);
		}
		else if (waveletType == "shannoncosine") {
			dwtShannonCosine(currentSrc, tempA, tempH, tempV, tempD);
		}
		else {
			throw std::invalid_argument("Unsupported wavelet type");
		}

		cA.push_back(tempA);
		cH.push_back(tempH);
		cV.push_back(tempV);
		cD.push_back(tempD);
		currentSrc = tempA; // Use approximation coefficients for the next level
	}
}

// Soft Thresholding
void softThreshold(cv::Mat& mat, float threshold) {
	for (int y = 0; y < mat.rows; ++y) {
		for (int x = 0; x < mat.cols; ++x) {
			float val = mat.at<float>(y, x);
			if (std::abs(val) <= threshold) {
				mat.at<float>(y, x) = 0;
			}
			else {
				mat.at<float>(y, x) = (val > 0 ? val - threshold : val + threshold);
			}
		}
	}
}

void thresholdCoefficients(std::vector<cv::Mat>& cH, std::vector<cv::Mat>& cV, std::vector<cv::Mat>& cD) {
	for (size_t i = 0; i < cH.size(); ++i) {
		float sigmaH = cv::mean(cv::abs(cH[i]))[0] / 0.6745;
		float sigmaV = cv::mean(cv::abs(cV[i]))[0] / 0.6745;
		float sigmaD = cv::mean(cv::abs(cD[i]))[0] / 0.6745;

		float thresholdH = sigmaH * std::sqrt(2 * std::log(cH[i].rows * cH[i].cols));
		float thresholdV = sigmaV * std::sqrt(2 * std::log(cV[i].rows * cV[i].cols));
		float thresholdD = sigmaD * std::sqrt(2 * std::log(cD[i].rows * cD[i].cols));

		softThreshold(cH[i], thresholdH);
		softThreshold(cV[i], thresholdV);
		softThreshold(cD[i], thresholdD);
	}
}

// Inverse Haar Transform
void idwtHaar(const cv::Mat& cA, const cv::Mat& cH, const cv::Mat& cV, const cv::Mat& cD, cv::Mat& dst) {
	int width = cA.cols * 2;
	int height = cA.rows * 2;

	dst = cv::Mat::zeros(height, width, CV_32F);

	for (int y = 0; y < cA.rows; ++y) {
		for (int x = 0; x < cA.cols; ++x) {
			float a = cA.at<float>(y, x);
			float h = cH.at<float>(y, x);
			float v = cV.at<float>(y, x);
			float d = cD.at<float>(y, x);

			dst.at<float>(y * 2, x * 2) = a + h + v + d;
			dst.at<float>(y * 2, x * 2 + 1) = a - h + v - d;
			dst.at<float>(y * 2 + 1, x * 2) = a + h - v - d;
			dst.at<float>(y * 2 + 1, x * 2 + 1) = a - h - v + d;
		}
	}
}

// Inverse Daubechies 4 Transform
void idwtDaubechies4(const cv::Mat& cA, const cv::Mat& cH, const cv::Mat& cV, const cv::Mat& cD, cv::Mat& dst) {
	int width = cA.cols * 2;
	int height = cA.rows * 2;

	dst = cv::Mat::zeros(height, width, CV_32F);

	for (int y = 0; y < cA.rows; ++y) {
		for (int x = 0; x < cA.cols; ++x) {
			float a = cA.at<float>(y, x);
			float h = cH.at<float>(y, x);
			float v = cV.at<float>(y, x);
			float d = cD.at<float>(y, x);

			dst.at<float>(y * 2, x * 2) = a * h0 + h * g0 + v * h0 + d * g0;
			dst.at<float>(y * 2, x * 2 + 1) = a * h1 + h * g1 + v * h1 + d * g1;
			dst.at<float>(y * 2 + 1, x * 2) = a * h2 + h * g2 + v * h2 + d * g2;
			dst.at<float>(y * 2 + 1, x * 2 + 1) = a * h3 + h * g3 + v * h3 + d * g3;
		}
	}
}

// Inverse Shannon–Cosine Transform
void idwtShannonCosine(const cv::Mat& cA, const cv::Mat& cH, const cv::Mat& cV, const cv::Mat& cD, cv::Mat& dst) {
	int width = cA.cols * 2;
	int height = cA.rows * 2;

	dst = cv::Mat::zeros(height, width, CV_32F);

	for (int y = 0; y < cA.rows; ++y) {
		for (int x = 0; x < cA.cols; ++x) {
			float a = cA.at<float>(y, x);
			float h = cH.at<float>(y, x);
			float v = cV.at<float>(y, x);
			float d = cD.at<float>(y, x);

			dst.at<float>(y * 2, x * 2) = a + h + v + d;
			dst.at<float>(y * 2, x * 2 + 1) = a - h + v - d;
			dst.at<float>(y * 2 + 1, x * 2) = a + h - v - d;
			dst.at<float>(y * 2 + 1, x * 2 + 1) = a - h - v + d;
		}
	}
}

// Multilevel Wavelet Reconstruction
void multilevelIDWT(const std::vector<cv::Mat>& cA, const std::vector<cv::Mat>& cH, const std::vector<cv::Mat>& cV, const std::vector<cv::Mat>& cD, cv::Mat& dst, const std::string& waveletType) {
	dst = cA.back();
	for (int i = cA.size() - 1; i >= 0; --i) {
		cv::Mat temp;
		if (waveletType == "haar") {
			idwtHaar(cA[i], cH[i], cV[i], cD[i], temp);
		}
		else if (waveletType == "daubechies4") {
			idwtDaubechies4(cA[i], cH[i], cV[i], cD[i], temp);
		}
		else if (waveletType == "shannoncosine") {
			idwtShannonCosine(cA[i], cH[i], cV[i], cD[i], temp);
		}
		else {
			throw std::invalid_argument("Unsupported wavelet type");
		}
		dst = temp.clone();
	}
}

int testDwt(std::string inputFile)
{
	// Read the input image
	Mat src = imread(inputFile, IMREAD_GRAYSCALE | IMREAD_ANYDEPTH);
	if (src.empty()) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	// Number of decomposition levels
	int levels = 5;

	// Wavelet type
	std::string waveletType = "shannoncosine"; // Change to "haar" or "daubechies4"

	// Vectors to hold the subbands for each level
	std::vector<cv::Mat> cA, cH, cV, cD;

	// Perform the multilevel wavelet transform
	multilevelDWT(src, cA, cH, cV, cD, levels, waveletType);

	// Perform soft thresholding on the detail coefficients
	thresholdCoefficients(cH, cV, cD);

	// Reconstruct the image from the thresholded coefficients
	cv::Mat reconstructed;
	multilevelIDWT(cA, cH, cV, cD, reconstructed, waveletType);

	// Convert the result to 8-bit and display
	cv::Mat reconstructed_8U;
	reconstructed.convertTo(reconstructed_8U, CV_16U);

	return 0;
}




// BidimensionalEmpiricalModeDecomnposition

// Function to find local maxima and minima using a more robust method
void findExtrema(const cv::Mat& image, cv::Mat& maxima, cv::Mat& minima) {

	cv::Mat dilatedMax, erodedMin;
	cv::dilate(image, dilatedMax, cv::Mat());
	cv::erode(image, erodedMin, cv::Mat());

	maxima = (image == dilatedMax);
	minima = (image == erodedMin);
}



void maskedFilterGauss(const cv::Mat& cvProj, const cv::Mat& cvSmaskData, cv::Mat& result, double sigma_px)
{
	cv::Mat binaryMask;
	cv::threshold(cvSmaskData, binaryMask, 0., 1, cv::THRESH_BINARY);
	cv::Mat gaussMask;
	cv::GaussianBlur(binaryMask, gaussMask, cv::Size(), sigma_px);

	cv::Mat projFiltered;
	cv::multiply(cvProj, binaryMask, projFiltered);
	cv::GaussianBlur(projFiltered, projFiltered, cv::Size(), sigma_px);

	gaussMask += std::numeric_limits<float>::epsilon();
	cv::divide(projFiltered, gaussMask, result);
}

void boxFilterMask(const cv::Mat& src, const cv::Mat& mask, cv::Mat& result, int maskSize)
{
	cv::Mat binaryMask;
	cv::threshold(mask, binaryMask, 0., 1, cv::THRESH_BINARY);
	binaryMask.convertTo(binaryMask, CV_32F);

	cv::Mat gaussMask;
	cv::boxFilter(binaryMask, gaussMask, -1, cv::Size(maskSize, maskSize));

	cv::Mat srcFiltered;
	cv::multiply(src, binaryMask, srcFiltered);
	cv::boxFilter(srcFiltered, srcFiltered, -1, cv::Size(maskSize, maskSize));

	gaussMask += std::numeric_limits<float>::epsilon();
	cv::divide(srcFiltered, gaussMask, result);
}

// Helper function to compute B-spline basis functions
std::vector<float> bsplineBasis(int k, float t, const std::vector<float>& knots) {
	std::vector<float> b(knots.size() - k - 1, 0.0f);
	if (k == 0) {
		for (int i = 0; i < b.size(); ++i) {
			if (t >= knots[i] && t < knots[i + 1]) {
				b[i] = 1.0f;
			}
		}
	}
	else {
		std::vector<float> b0 = bsplineBasis(k - 1, t, knots);
		for (int i = 0; i < b.size(); ++i) {
			float denom1 = knots[i + k] - knots[i];
			float term1 = denom1 != 0 ? ((t - knots[i]) / denom1) * b0[i] : 0;
			float denom2 = knots[i + k + 1] - knots[i + 1];
			float term2 = denom2 != 0 ? ((knots[i + k + 1] - t) / denom2) * b0[i + 1] : 0;
			b[i] = term1 + term2;
		}
	}
	return b;
}

cv::Mat interpolateSpline(const std::vector<cv::Point>& points, const cv::Mat& values, const cv::Size& size, int k = 3) {
	int n = points.size();
	std::vector<float> knots(n + k + 1, 0.0f);
	for (int i = 0; i <= n + k; ++i) {
		knots[i] = i < k + 1 ? 0.0f : (i > n ? n - k : i - k);
	}

	cv::Mat interpolated(size, CV_32F, cv::Scalar(0));
	cv::Mat count(size, CV_32F, cv::Scalar(0));

	for (int y = 0; y < size.height; ++y) {
		for (int x = 0; x < size.width; ++x) {
			float t = (float)x / (size.width - 1) * (n - k);
			std::vector<float> basis = bsplineBasis(k, t, knots);

			for (int i = 0; i < n; ++i) {
				interpolated.at<float>(y, x) += basis[i] * values.at<float>(points[i]);
				count.at<float>(y, x) += basis[i];
			}
		}
	}

	for (int y = 0; y < size.height; ++y) {
		for (int x = 0; x < size.width; ++x) {
			if (count.at<float>(y, x) > 0) {
				interpolated.at<float>(y, x) /= count.at<float>(y, x);
			}
		}
	}

	return interpolated;
}


// Function to create envelopes using simplified B-spline interpolation
void createEnvelopes(const cv::Mat& image, const cv::Mat& extrema, cv::Mat& envelope, int maskSize) {
	std::vector<cv::Point> points;
	cv::Mat values(extrema.size(), CV_32F, cv::Scalar(0));

	for (int y = 0; y < extrema.rows; ++y) {
		for (int x = 0; x < extrema.cols; ++x) {
			if (extrema.at<uchar>(y, x) == 255) {
				points.emplace_back(x, y);
				values.at<float>(y, x) = image.at<float>(y, x);
			}
		}
	}

	//envelope = interpolateSpline(points, values, image.size());
	// maskedFilterGauss(values, values, envelope, maskSize);

	/*cv::Mat imageFloat;
	image.convertTo(imageFloat, CV_32F);
	envelope = cv::Mat::zeros(image.size(), CV_32F);*/
	envelope = interpolateSpline(points, values, image.size());
}

// Function to perform the sifting process
void sift(const cv::Mat& image, cv::Mat& IMF, int maskSize) {
	cv::Mat maxima, minima, upperEnvelope, lowerEnvelope, meanEnvelope;
	findExtrema(image, maxima, minima);
	createEnvelopes(image, maxima, upperEnvelope, maskSize);
	createEnvelopes(image, minima, lowerEnvelope, maskSize);

	// Calculate the mean envelope
	meanEnvelope = (upperEnvelope + lowerEnvelope) / 2;

	// Subtract the mean envelope from the original image
	IMF = image - meanEnvelope;
}


int BMED(std::string inputFile, std::string inputFlatFile, std::string outputFile)
{
	//load image
	Mat image = imread(inputFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);
	Mat flat = imread(inputFlatFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);

	image.convertTo(image, CV_32F);
	cv::log(image, image);

	flat.convertTo(flat, CV_32F);
	cv::log(flat, flat);

	image = flat - image;

	// Step 2: Initialize parameters
	int maxIterations = 10; // Max number of iterations for sifting
	std::vector<cv::Mat> IMFs;


	// Step 3: Perform EMD
	int maskSize = 20;
	cv::Mat residue = image.clone();
	for (int i = 0; i < 4; ++i)
		cv::pyrDown(residue, residue);

	for (int i = 0; i < maxIterations; ++i) {
		cv::Mat IMF;
		sift(residue, IMF, maskSize * (i + 1) + 1);
		IMFs.push_back(IMF);
		residue -= IMF;
		cv::pyrDown(residue, residue);

		// Termination criteria: residue becomes small or no more extrema
		double minVal, maxVal;
		cv::minMaxLoc(residue, &minVal, &maxVal);
		if (maxVal - minVal < 0.1) {
			break;
		}
	}

	// Step 4: Display results
	cv::imshow("Original Image", image);
	for (size_t i = 0; i < IMFs.size(); ++i) {
		std::string windowName = "IMF " + std::to_string(i + 1);
		cv::imshow(windowName, IMFs[i]);
	}
	cv::imshow("Residue", residue);

	cv::waitKey(0);
	return 0;
}






#if 0

int main()
{
	std::string inputBMED = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\data\\04\\png_reduced\\04_06.png";
	std::string inputFlatBMED = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\Flat_field\\31kVp-120mAs- saturata\\png_reduced_80mAsTo120mAs\\07.png";
	std::string outputFileBMED = "d:\\tmp\\outBMED.tif";

	BMED(inputBMED, inputFlatBMED, outputFileBMED);


#if 0
	std::string inputFileDWT = "E:\\temp\\GaussianFourierPyramid\\GaussianFourierPyramid\\04\\SIRT_reduced_NoDen\\slices\\s_0025.png";
	testDwt(inputFileDWT);
#endif


#if 0
	// bilateral MSE
	//std::string inputFile = ".\\s_0027.png";
	std::string inputFile = "C:\\Users\\jack\\OneDrive - digitecinnovation.com\\imagesSamples\\png_reduced\\06_06.png";
	std::string outputFile = ".\\s_0027_bilateral.png";
	testBilateralDenoise(inputFile, outputFile);
#endif
	// bilateral MSE
	//std::string inputFile = ".\\s_0027.png";
	std::string inputFileT0 = "E:\\temp\\GaussianFourierPyramid\\GaussianFourierPyramid\\04\\SIRT_reduced_NoDen\\slices\\s_0025.png";
	std::string outputFileT0 = ".\\s_0025_bilateral.png";
	testDoubleBilateral(inputFileT0, outputFileT0);


	// enh slice 2D
	std::string inputFolderEnhSlice = "E:\\temp\\GaussianFourierPyramid\\GaussianFourierPyramid\\04\\SIRT_reduced_NoDen\\slices";
	std::string outputFolderEnhSlice = ".\\outEnhSlice04";

	if (!std::filesystem::exists(outputFolderEnhSlice))
		std::filesystem::create_directories(outputFolderEnhSlice);

	testVolumeConspicuity(inputFolderEnhSlice, outputFolderEnhSlice);


	// gamma enh volume
	std::string inputFolderGammaAdaptive = "E:\\temp\\GaussianFourierPyramid\\GaussianFourierPyramid\\14\\SIRT_reduced\\slices";
	std::string outputFolderGammaAdaptive = ".\\outGammaAdaptive";

	if (!std::filesystem::exists(outputFolderGammaAdaptive))
		std::filesystem::create_directories(outputFolderGammaAdaptive);

	const ushort maskThreshold14{ 500 };
	testAdaptiveGammaVolume(inputFolderGammaAdaptive, outputFolderGammaAdaptive, maskThreshold14);




	// equalize proj
	const ushort maskThresholdEQ{ 300 };
	std::string inputEQ = "d:\\tmp\\cvProj.tif";
	std::string outputEQ = "d:\\tmp\\cvProjEq.tif";
	testEqualize(inputEQ, outputEQ, maskThresholdEQ);


	std::string inputFolderMSEX = "d:\\agfa";
	std::string outputFolderMSEX = ".\\outAgfa";

	if (!std::filesystem::exists(outputFolderMSEX))
		std::filesystem::create_directories(outputFolderMSEX);
	testMseEnhProjWithLog(inputFolderMSEX, outputFolderMSEX);


	const ushort maskThreshold{ 500 };
	//std::string inputTB = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\data\\01\\png_reduced\\01_06.png";
	//std::string inputFlatTB = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\Flat_field\\29kVp-55mAs\\png_reduced\\07.png";
	//std::string outputFileTB = ".\\outBand.tif";

	//std::string inputTB = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\data\\08\\png_reduced\\08_06.png";
	//std::string inputFlatTB = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\Flat_field\\29kVp-81mAs\\png_reduced\\07.png";
	std::string inputTB = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\data\\04\\png_reduced\\04_06.png";
	std::string inputFlatTB = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\Flat_field\\31kVp-120mAs- saturata\\png_reduced\\07.png";
	std::string outputFileTB = ".\\outBand.tif";

	int distThresholdTB{ 400 };
	testBand(inputTB, inputFlatTB, outputFileTB, maskThreshold, distThresholdTB);


	std::string inputLR = "E:\\soliddetectorimages\\TomoImages\\output\\20240105 Metaltronica\\06\\SIRT_reduced\\Synth2DProjection.png";
	std::string outputLR = ".\\outLR.tif";
	testLR(inputLR, outputLR);




	std::string inputFolderDenoise = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\data\\01\\png_reduced";
	std::string inputFlat = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\Flat_field\\29kVp-55mAs\\png_reduced\\07.png";
	std::string outputFolderDenoise = ".\\outDenoise";

	if (!std::filesystem::exists(outputFolderDenoise))
		std::filesystem::create_directories(outputFolderDenoise);

	testprojDenoise(inputFolderDenoise, inputFlat, outputFolderDenoise);



	// mse enh volume
	//std::string inputFolder = "C:\\Users\\jack\\OneDrive - digitecinnovation.com\\imagesSamples\\slice 08\\slices";
	std::string inputFolder1 = "E:\\temp\\GaussianFourierPyramid\\GaussianFourierPyramid\\04\\SIRT_reduced_NoDen\\slices";
	std::string outputFolder1 = ".\\outSigma04";

	if (!std::filesystem::exists(outputFolder1))
		std::filesystem::create_directories(outputFolder1);

	testMseEnhVolume(inputFolder1, outputFolder1);

	// gamma enh volume
	std::string inputFolderGamma1 = ".\\outSigma04";
	std::string outputFolderGamma1 = ".\\outSigma04Gamma";

	if (!std::filesystem::exists(outputFolderGamma1))
		std::filesystem::create_directories(outputFolderGamma1);

	testSigmoidVolume(inputFolderGamma1, outputFolderGamma1, maskThreshold);


	// bilateral MSE
	//std::string inputFile = ".\\s_0027.png";
	std::string inputFile2D = "C:\\Users\\jack\\OneDrive - digitecinnovation.com\\imagesSamples\\synthetic_proj_muTau-exm-1-proj_7.tif";
	std::string outputFile2D = ".\\synthetic_proj_muTau-exm-1-proj_7_enh.tif";
	test2DSynth(inputFile2D, outputFile2D, double(maskThreshold));

	// bilateral MSE
	//std::string inputFile = ".\\s_0027.png";
	std::string inputFile = "C:\\Users\\jack\\OneDrive - digitecinnovation.com\\imagesSamples\\png_reduced\\06_06.png";
	std::string outputFile = ".\\s_0027_bilateral.png";
	testBilateralDenoise(inputFile, outputFile);

	std::string inputFileVolSlide = "C:\\Users\\jack\\OneDrive - digitecinnovation.com\\imagesSamples\\synthetic_proj_muTau-exm-8-proj_7.png";
	std::string outputFileVolSlide = ".\\synthetic_proj_muTau-exm-8-proj_7.tif";
	testBilateralHPF(inputFileVolSlide, outputFileVolSlide);


	// gamma
	std::string inputFileGamma = ".\\s_0027.png";
	std::string outputFileGamma = ".\\s_0027_gamma.png";
	testGamma(inputFileGamma, outputFileGamma, maskThreshold);


	// mse enh volume
	//std::string inputFolder = "C:\\Users\\jack\\OneDrive - digitecinnovation.com\\imagesSamples\\slice 08\\slices";
	std::string inputFolder = "E:\\temp\\GaussianFourierPyramid\\GaussianFourierPyramid\\04\\SIRT_reduced_NoDen\\slices";
	std::string outputFolder = ".\\outSigma04";

	if (!std::filesystem::exists(outputFolder))
		std::filesystem::create_directories(outputFolder);

	testMseEnhVolume(inputFolder, outputFolder);


	// gamma enh volume
	std::string inputFolderGamma = "C:\\Users\\jack\\OneDrive - digitecinnovation.com\\imagesSamples\\slice 08\\slices";
	std::string outputFolderGamma = ".\\outSigma50Gamma";

	if (!std::filesystem::exists(outputFolderGamma))
		std::filesystem::create_directories(outputFolderGamma);

	testGammaVolume(inputFolderGamma, outputFolderGamma, maskThreshold);

	// gamma enh volume
	std::string inputFolderGammaMSE = ".\\outSigma50MSE";
	std::string outputFolderGammaMSE = ".\\outSigma50MSEGamma";

	if (!std::filesystem::exists(outputFolderGammaMSE))
		std::filesystem::create_directories(outputFolderGammaMSE);

	testGammaVolume(inputFolderGammaMSE, outputFolderGammaMSE, maskThreshold);

	return 0;
}

#endif

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tinynurbs/tinynurbs.h"

// Define a structure to hold the points
struct tPoint {
	double x, y, z;
};

// Function to create a NURBS curve
tinynurbs::RationalCurve3d createNURBSCurve(const std::vector<tPoint>& points, int degree) {
	tinynurbs::RationalCurve3d curve;
	curve.degree = degree;

	for (const auto& point : points) {
		curve.control_points.push_back(glm::vec3(point.x, point.y, point.z));
		curve.weights.push_back(1.0); // uniform weights
	}

	// Generate knot vector
	curve.knots = tinynurbs::knotvector::uniform(points.size(), degree);

	return curve;
}

// Function to evaluate the NURBS curve on a regular grid and generate an OpenCV image
cv::Mat evaluateNURBSOnGrid(const tinynurbs::RationalCurve3d& curve, int gridSize) {
	cv::Mat image(gridSize, gridSize, CV_8UC1, cv::Scalar(0));

	for (int i = 0; i < gridSize; ++i) {
		for (int j = 0; j < gridSize; ++j) {
			double u = double(i) / (gridSize - 1);

			glm::vec3 result = tinynurbs::curvePoint(curve, u);

			int intensity = static_cast<int>(result.z * 255); // Scale z-value to 0-255 range
			intensity = std::max(0, std::min(255, intensity));  // Clamp values to 0-255
			image.at<uchar>(i, j) = intensity;
		}
	}

	return image;
}

int main() {
	// Example set of points
	std::vector<tPoint> points = {
		{0.1, 0.2, 1.0},
		{0.4, 0.5, 2.0},
		{0.6, 0.7, 3.0},
		{0.9, 0.8, 4.0}
	};

	int degree = 3;
	tinynurbs::RationalCurve3d curve = createNURBSCurve(points, degree);

	int gridSize = 512;
	cv::Mat image = evaluateNURBSOnGrid(curve, gridSize);

	// Display the image
	cv::imshow("NURBS Interpolation", image);
	cv::waitKey(0);

	return 0;
}



#if 0


#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "tinynurbs/tinynurbs.h"

// Define a structure to hold the points
struct Point {
	double x, y, z;
};

// Function to create a NURBS curve
tinynurbs::RationalCurve3d createNURBSCurve(const std::vector<Point>& points, int degree) {
	tinynurbs::RationalCurve3d curve;
	curve.degree = degree;

	for (const auto& point : points) {
		curve.control_points.push_back(glm::vec3(point.x, point.y, point.z));
		curve.weights.push_back(1.0); // uniform weights
	}

	// Generate knot vector
	curve.knots = tinynurbs::knotvector::uniform(points.size(), degree);

	return curve;
}

// Function to evaluate the NURBS curve on a regular grid and generate an OpenCV image
cv::Mat evaluateNURBSOnGrid(const tinynurbs::RationalCurve3d& curve, int gridSize) {
	cv::Mat image(gridSize, gridSize, CV_8UC1, cv::Scalar(0));

	for (int i = 0; i < gridSize; ++i) {
		for (int j = 0; j < gridSize; ++j) {
			double u = double(i) / (gridSize - 1);

			glm::vec3 result = tinynurbs::curvePoint(curve, u);

			int intensity = static_cast<int>(result.z * 255); // Scale z-value to 0-255 range
			intensity = std::max(0, std::min(255, intensity));  // Clamp values to 0-255
			image.at<uchar>(i, j) = intensity;
		}
	}

	return image;
}

int main() {
	// Example set of points
	std::vector<Point> points = {
		{0.1, 0.2, 1.0},
		{0.4, 0.5, 2.0},
		{0.6, 0.7, 3.0},
		{0.9, 0.8, 4.0}
	};

	int degree = 3;
	tinynurbs::RationalCurve3d curve = createNURBSCurve(points, degree);

	int gridSize = 512;
	cv::Mat image = evaluateNURBSOnGrid(curve, gridSize);

	// Display the image
	cv::imshow("NURBS Interpolation", image);
	cv::waitKey(0);

	return 0;
}

#endif
