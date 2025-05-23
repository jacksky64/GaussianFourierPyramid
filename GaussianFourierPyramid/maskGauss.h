#include <opencv2/core.hpp>

cv::Mat maskedGaussianGPU(const cv::Mat& grayscale, const cv::Mat& mask, const cv::Mat& contour, float sigma);
