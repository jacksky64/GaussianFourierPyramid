#include "LLF.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

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

float bilinearInterpolate(const Mat& image, Point2f p)
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
	float I11 = image.at<float>(y1, x1);
	float I12 = image.at<float>(y2, x1);
	float I21 = image.at<float>(y1, x2);
	float I22 = image.at<float>(y2, x2);

	// Perform bilinear interpolation
	float I = (1 - a) * (1 - b) * I11 +
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

		Point2f newPoint{ float(point.x),float(point.y) };
		currentPath.push_back(newPoint);

		while ((distance.at<float>(cvRound(newPoint.y), cvRound(newPoint.x)) < thresholdDist) && !borderCollision)
		{
			int x = cvRound(newPoint.x);
			int y = cvRound(newPoint.y);
			float dx = (float)gradX.at<double>(y, x);
			float dy = (float)gradY.at<double>(y, x);
			Point2f t = Point2f(((float)x + dx * contractionStep), ((float)y + dy * contractionStep));

			x = cvRound(t.x);
			y = cvRound(t.y);
			if (x >= 0 && x < gradX.cols && y >= 0 && y < gradX.rows)
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

		warpPerspective(cvDistance, r, perspectiveMatrix, r.size(), INTER_LINEAR, BORDER_TRANSPARENT);

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
	srcCropped = src.colRange(Range(450, src.cols - 1));
	// anatomic mask
	cv::Mat anatomicMask;
	cv::compare(srcCropped, maskThreshold, anatomicMask, cv::CmpTypes::CMP_GT);

	// Debug 


		// MSE filter
	MSEGaussRemap mse;
	Mat filtered;
	const float sigmaMSE = 100.f;
	const std::vector<float> boostMSE{ 1.f,3.f,3.f,3.f,3.f,3.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f,-1.f };
	const int levelMSE = 10;
	mse.filter(srcCropped, filtered, sigmaMSE, boostMSE, levelMSE);


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




int main()
{
	const ushort maskThreshold{ 500 };

	int distThreshold{ 300 };
	//std::string inputTB = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\data\\01\\png_reduced\\01_06.png";
	//std::string inputFlatTB = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\Flat_field\\29kVp-55mAs\\png_reduced\\07.png";
	//std::string outputFileTB = ".\\outBand.tif";

	std::string inputTB = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\data\\08\\png_reduced\\08_06.png";
	std::string inputFlatTB = "E:\\soliddetectorimages\\TomoImages\\20240105 Metaltronica\\Flat_field\\29kVp-81mAs\\png_reduced\\07.png";
	std::string outputFileTB = ".\\outBand.tif";
	testBand(inputTB, inputFlatTB, outputFileTB, maskThreshold, distThreshold);


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