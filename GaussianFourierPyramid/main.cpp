#include "LLF.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

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
	mse.filter(dn, LaplacianLevels, sigmaRangeBilateral, sigmaSpaceBilateral, filterSize, levelMSE);

	// invert VST
	pow(LaplacianLevels, 2., LaplacianLevels);
	LaplacianLevels.convertTo(LaplacianLevels, CV_16U);

	cv::imwrite(outputFile, LaplacianLevels);

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
#pragma endregion

#pragma region MSEblend
//
//class MSEBlend
//{
//	// Function to build Gaussian pyramid
//	void buildGaussianPyramid(const Mat& src, vector<Mat>& pyramid, int levels)
//	{
//		pyramid.push_back(src);
//		Mat current = src;
//		for (int i = 1; i < levels; ++i) {
//			Mat down;
//			pyrDown(current, down);
//			pyramid.push_back(down);
//			current = down;
//		}
//	}
//
//	// Function to build Laplacian pyramid
//	void buildLaplacianPyramid(const vector<Mat>& gaussPyr, vector<Mat>& laplPyr)
//	{
//		for (size_t i = 0; i < gaussPyr.size() - 1; ++i) {
//			Mat up;
//			pyrUp(gaussPyr[i + 1], up, gaussPyr[i].size());
//			Mat lap = gaussPyr[i] - up;
//			laplPyr.push_back(lap);
//		}
//		laplPyr.push_back(gaussPyr.back()); // The smallest level is the same as in the Gaussian pyramid
//	}
//
//	// Function to blend Laplacian pyramids
//	vector<Mat> blendPyramids(const vector<Mat>& laplPyr1, const vector<Mat>& laplPyr2, const vector<Mat>& gaussPyrMask)
//	{
//		vector<Mat> blendedPyr;
//		for (size_t i = 0; i < laplPyr1.size(); ++i) {
//			Mat blended = laplPyr1[i].mul(gaussPyrMask[i]) + laplPyr2[i].mul(Scalar::all(1.0) - gaussPyrMask[i]);
//			blendedPyr.push_back(blended);
//		}
//		return blendedPyr;
//	}
//
//	// Function to reconstruct image from Laplacian pyramid
//	Mat reconstructFromLaplacianPyramid(const vector<Mat>& laplPyr)
//	{
//		Mat current = laplPyr.back();
//		for (size_t i = laplPyr.size() - 2; i < laplPyr.size(); --i) 
//		{
//			Mat up;
//			pyrUp(current, up, laplPyr[i].size());
//			current = up + laplPyr[i];
//		}
//		return current;
//	}
//
//	int testMSEBlend()
//	{
//		std::string src1;
//		std::string src2;
//		std::string msk;
//
//		// Load the images
//		Mat img1 = imread(src1, IMREAD_COLOR);
//		Mat img2 = imread(src2, IMREAD_COLOR);
//		Mat mask = imread(msk, IMREAD_GRAYSCALE);
//
//		if (img1.empty() || img2.empty() || mask.empty()) {
//			return -1;
//		}
//
//		if (img1.size() != img2.size() || img1.size() != mask.size()) {
//			return -1;
//		}
//
//		// Convert mask to float and normalize to [0, 1]
//		mask.convertTo(mask, CV_32F, 1.0 / 255.0);
//
//		// Number of pyramid levels
//		int levels = 6;
//
//		// Build Gaussian pyramids
//		vector<Mat> gaussPyr1, gaussPyr2, gaussPyrMask;
//		buildGaussianPyramid(img1, gaussPyr1, levels);
//		buildGaussianPyramid(img2, gaussPyr2, levels);
//		buildGaussianPyramid(mask, gaussPyrMask, levels);
//
//		// Build Laplacian pyramids
//		vector<Mat> laplPyr1, laplPyr2;
//		buildLaplacianPyramid(gaussPyr1, laplPyr1);
//		buildLaplacianPyramid(gaussPyr2, laplPyr2);
//
//		// Blend Laplacian pyramids
//		vector<Mat> blendedPyr = blendPyramids(laplPyr1, laplPyr2, gaussPyrMask);
//
//		// Reconstruct the blended image from the Laplacian pyramid
//		Mat blendedImage = reconstructFromLaplacianPyramid(blendedPyr);
//
//		// Save and display the result
//		imwrite("blended_image.png", blendedImage);
//
//		return 0;
//	}
//};
#pragma endregion



int main()
{
	const ushort maskThreshold{ 500 };

	// bilateral MSE
	//std::string inputFile = ".\\s_0027.png";
	std::string inputFile = "C:\\Users\\jack\\OneDrive - digitecinnovation.com\\imagesSamples\\png_reduced\\06_06.png";
	std::string outputFile = ".\\s_0027_bilateral.png";
	testBilateralDenoise(inputFile, outputFile);

	std::string inputFileVolSlide = "C:\\Users\\jack\\OneDrive - digitecinnovation.com\\imagesSamples\\slice_012_1.png";
	std::string outputFileVolSlide = ".\\slice_012_1_bilateral.png";
	testBilateralHPF(inputFileVolSlide, outputFileVolSlide);


	// gamma
	std::string inputFileGamma = ".\\s_0027.png";
	std::string outputFileGamma = ".\\s_0027_gamma.png";
	testGamma(inputFileGamma, outputFileGamma, maskThreshold);


	// mse enh volume
	std::string inputFolder = "C:\\Users\\jack\\OneDrive - digitecinnovation.com\\imagesSamples\\slice 08\\slices";
	std::string outputFolder = ".\\outSigma50MSE";

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