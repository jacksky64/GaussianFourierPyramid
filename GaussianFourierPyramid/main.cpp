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

#pragma region gammaEnh
class gammaEnh
{
public:
	void applyGamma(const cv::Mat& srcDest, const cv::Mat& mask, double gamma, double tailPercent = 0.001, double enlargeRangePerc = 0.1)
	{	
		ushort minValue{ 0 };
		ushort maxValue{ 0 };

		std::vector<cv::Mat> s{ srcDest };
		std::vector<cv::Mat> m{ mask};

		findMinMax(s, m, minValue, maxValue, tailPercent);

		// enlarge range
		const double range{ double(maxValue) - double(minValue) };
		const double rangeAdj{ range * enlargeRangePerc };

		const double enlargedMinValue = std::max(0., double(minValue) - rangeAdj);
		const double enlargedMaxValue = std::min(65535., double(maxValue) + rangeAdj);
		const double enlargedRange{ enlargedMaxValue - enlargedMinValue };

		gammaCorrection(s, gamma, ushort(enlargedMinValue), ushort(enlargedMaxValue));
	}

	void applyGamma(const std::vector<cv::Mat>& srcDest, const std::vector<cv::Mat>& mask, double gamma, double tailPercent = 0.001, double enlargeRangePerc = 0.1)
	{
		ushort minValue{ 0 };
		ushort maxValue{ 0 };

		findMinMax(srcDest, mask, minValue, maxValue, tailPercent);

		// enlarge range
		const double range{ double(maxValue) - double(minValue) };
		const double rangeAdj{ range * enlargeRangePerc };

		const double enlargedMinValue = std::max(0., double(minValue) - rangeAdj);
		const double enlargedMaxValue = std::min(65535., double(maxValue) + rangeAdj);
		const double enlargedRange{ enlargedMaxValue - enlargedMinValue };

		gammaCorrection(srcDest, gamma, ushort(enlargedMinValue), ushort(enlargedMaxValue));
	}

private:
	// Function to compute histogram and determine minValue and maxValue
	void findMinMax(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& mask, ushort& minValue, ushort& maxValue, double tailPercent = 0.01)
	{
		CV_Assert(src.size() == mask.size());

		// Calculate the histogram
		const int histSize = 65536;
		cv::Mat hist;
		float range[] = { 0, 65536 };
		const float* histRange = { range };
		bool uniform = true, accumulate = true;

		for (auto n = 0; n < src.size(); ++n)
		{
			auto s{ src.at(n)};
			auto m{ mask.at(n)};
			CV_Assert(s.type() == CV_16U);
			CV_Assert(m.type() == CV_8U);
			cv::calcHist(&s, 1, 0, m, hist, 1, &histSize, &histRange, uniform, accumulate);
		}

		// Calculate the cumulative histogram
		std::vector<int> cumulative(histSize, 0);
		cumulative[0] = (int)hist.at<float>(0);
		for (int i = 1; i < histSize; ++i) {
			cumulative[i] = cumulative[i - 1] + (int)hist.at<float>(i);
		}

		// Determine total number of pixels
		int totalPixels = cumulative[histSize - 1];

		// Determine the cutoff values
		int lowerCutoff = int(double(totalPixels) * tailPercent);
		int upperCutoff = int(double(totalPixels) * (1 - tailPercent));

		// Find minValue
		for (int i = 0; i < histSize; ++i) {
			if (cumulative[i] > lowerCutoff) {
				minValue = i;
				break;
			}
		}

		// Find maxValue
		for (int i = histSize - 1; i >= 0; --i) {
			if (cumulative[i] < upperCutoff) {
				maxValue = i;
				break;
			}
		}
	}

	// Function to apply gamma correction
	void gammaCorrection(const std::vector<cv::Mat>& src, double gamma, ushort minValue, ushort maxValue)
	{
		Mat lut = buildGammaLut(gamma, minValue, maxValue);
		const ushort* lut_data = lut.ptr<ushort>();

		for (int nImage=0; nImage<src.size();nImage++)
		{
			Mat s(src[nImage]);

			CV_Assert(!s.empty() );
			CV_Assert(s.type() == CV_16UC1);

			for (int y = 0; y < s.rows; ++y)
			{
				for (int x = 0; x < s.cols; ++x)
				{
					s.at<ushort>(y, x) = lut_data[s.at<ushort>(y, x)];
				}
			}
		}
	}

	Mat buildGammaLut(double gamma, ushort minValue, ushort maxValue)
	{
		CV_Assert(gamma >= 0);
		CV_Assert(minValue < maxValue);

		// Calculate the scale and offset
		double scale = 1.0 / (maxValue - minValue);
		double offset = minValue;

		// Create a look-up table
		cv::Mat lut(1, 65536, CV_16UC1);
		ushort* lut_data = lut.ptr<ushort>();

		// gamma between minValue and maxValue and linear outside
		for (int i = 0; i < 65536; ++i) {
			if (i < minValue) {
				lut_data[i] = i;
			}
			else if (i > maxValue) {
				lut_data[i] = i;
			}
			else {
				double normalized = (i - offset) * scale;
				lut_data[i] = cv::saturate_cast<ushort>(pow(normalized, gamma) * (maxValue - minValue) + minValue);
			}
		}

		return lut;
	}

};
#pragma endregion

//alpha blending comparison between src1 and src2 by GUI
void compare(const string wname, const Mat& src1, const Mat& src2)
{
	namedWindow(wname);
	int a = 0; createTrackbar("alpha", wname, &a, 100);
	int key = 0;
	Mat show;
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

		Mat zoomedSrc1;
		resize(src1, zoomedSrc1, Size(), zoomLevel, zoomLevel);
		Mat zoomedSrc2;
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
static std::vector<Mat> mseEnhVolume(std::vector<Mat>& inputSlices, float sigmaMSE, const std::vector<float>& boostMSE, int levelMSE)
{
	// MSE filter
	MSEGaussRemap mse;

	std::vector<Mat> enhSlices;
	for (const auto& slice : inputSlices)
	{
		Mat filtered;

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

	std::vector<Mat> inputSlices;
	for (const auto& fn : inputFiles)
		inputSlices.push_back(imread(fn, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH));

	//parameter setting
	const float sigmaMSE = 100.f;	
	const std::vector<float> boostMSE{ 1.f, 1.2f };
	const int levelMSE = 6;

	std::vector<Mat> enhSlices = mseEnhVolume(inputSlices, sigmaMSE, boostMSE, levelMSE);

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

int testBilateral(std::string& inputFile, std::string& outputFile)
{
	//load image
	cv::Mat srcOriginal;
	srcOriginal = imread(inputFile, cv::IMREAD_GRAYSCALE | cv::IMREAD_ANYDEPTH);


	//parameter setting
	const float sigmaRangeBilateral = 0.35f;
	const float sigmaSpaceBilateral = 1.5f;
	const int levelMSE = 2;
	const int filterSize = 15;

	Mat dn;
	srcOriginal.convertTo(dn, CV_32F);
	sqrt(dn, dn);

	MSEBilateral mse;
	Mat LaplacianLevels;

	// MSE - gamma
	mse.filter(dn, LaplacianLevels, sigmaRangeBilateral, sigmaSpaceBilateral, filterSize, levelMSE);

	pow(LaplacianLevels, 2., LaplacianLevels );
	LaplacianLevels.convertTo(LaplacianLevels, CV_16U);
	
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
		cv::compare(slices[slices.size()-1], maskThreshold, anatomicMask, cv::CmpTypes::CMP_GT);
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


#pragma endregion


int main()
{
	const ushort maskThreshold{ 500 };

	// bilateral MSE
	//std::string inputFile = ".\\s_0027.png";
	std::string inputFile = "C:\\Users\\jack\\OneDrive - digitecinnovation.com\\imagesSamples\\png_reduced\\06_06.png";
	std::string outputFile = ".\\s_0027_bilateral.png";
	testBilateral(inputFile, outputFile);


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