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
	mse.filter(dn, LaplacianLevels, std::vector<float> {sigmaRangeBilateral},sigmaSpaceBilateral, filterSize, levelMSE);

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

	const int levels = 9;
	const float sigmaSpace{ 100 };
	filter.filter(src, result, sigmaSpace, std::vector<float>{0.,0.,0., 0., 0., 0., 0., -0.8, -0.9}, levels);

	cv::imwrite(outputFile, result);

	return 0;
}

#pragma endregion



int main()
{
	const ushort maskThreshold{ 500 };

	std::string inputLR = "\\\\dgtnas\\AImages\\Tomosynthesis\\output\\20240105 Metaltronica\\__28-05-2024_Issue_49344\\05\\SIRT_reduced\\Synth2DProjection.png";
	std::string outputLR = ".\\outLR.tif";
	testLR(inputLR,outputLR);


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