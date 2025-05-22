#include <opencv2/core.hpp>

bool computeWeightedAverage(const uchar* mask, const cv::Point* contourPoints, const float* contourValues,
    float* output, int rows, int cols, int numContourPoints, float sigma);
