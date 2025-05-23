#include "maskGauss.h"

#include <opencv2\cudaimgproc.hpp>
#include <cuda.h>
#include <cuda_runtime.h> // Ensure this header is included for CUDA math functions

#include <vector>
#include <iostream>


__global__ void computeWeightedAverageKernel(const uchar* mask, const cv::Point* contourPoints, const float* contourValues,
	float* output, int rows, int cols, int numContourPoints, float sigma2)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < cols && y < rows && mask[y * cols + x] != 0)
	{
		float weightedSum = 0.0;
		float weightTotal = 0.0;

		for (int n = 0; n < numContourPoints; ++n)
		{
			const cv::Point& cp = contourPoints[n];
			const float distance2 = (float(x - cp.x) * float(x - cp.x) + float(y - cp.y) * float(y - cp.y));
			const float weight = exp(-(distance2 / (2.f * sigma2)));
			//const float weight = 1.f / (1 + (distance2 / sigma2));
			const float intensity = contourValues[n];

			weightedSum += intensity * weight;
			weightTotal += weight;
		}

		output[y * cols + x] = (weightTotal > 0) ? weightedSum / weightTotal : 0.f;
	}
}

//bool computeWeightedAverage(const uchar* mask, const cv::Point* contourPoints, const float* contourValues,
//	float* output, int rows, int cols, int numContourPoints, float sigma)
//{
//	uchar* d_mask;
//	cv::Point* d_contourPoints;
//	float* d_contourValues;
//	float* d_output;
//
//	size_t maskSize = rows * cols * sizeof(uchar);
//	size_t pointSize = numContourPoints * sizeof(cv::Point);
//	size_t valueSize = numContourPoints * sizeof(float);
//	size_t outputSize = rows * cols * sizeof(float);
//
//	cudaMallocManaged(&d_mask, maskSize);
//	cudaMallocManaged(&d_contourPoints, pointSize);
//	cudaMallocManaged(&d_contourValues, valueSize);
//	cudaMallocManaged(&d_output, outputSize);
//
//	cudaMemcpy(d_mask, mask, maskSize, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_contourPoints, contourPoints, pointSize, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_contourValues, contourValues, valueSize, cudaMemcpyHostToDevice);
//
//	dim3 blockSize(32, 32);
//	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
//
//	// Ensure that gridSize.x and gridSize.y are within valid limits
//	if (gridSize.x > 65535 || gridSize.y > 65535) {
//		std::cout << "Grid size exceeds maximum limit." << std::endl;
//		return false;
//	}
//
//	cudaDeviceProp prop;
//	cudaGetDeviceProperties(&prop, 0);
//	if (blockSize.x * blockSize.y * blockSize.z > (size_t)prop.maxThreadsPerBlock)
//	{
//		std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
//		cudaFree(d_mask);
//		cudaFree(d_contourPoints);
//		cudaFree(d_contourValues);
//		cudaFree(d_output);
//		return false;
//	}
//
//	computeWeightedAverageKernel << <gridSize, blockSize >> > (d_mask, d_contourPoints, d_contourValues, d_output,
//		rows, cols, numContourPoints, sigma*sigma);
//
//	cudaError_t e = cudaGetLastError();
//	if (e != cudaSuccess)
//	{
//		std::cout << "CUDA Launch Error: " << cudaGetErrorString(e) << "\n";
//		cudaFree(d_mask);
//		cudaFree(d_contourPoints);
//		cudaFree(d_contourValues);
//		cudaFree(d_output);
//		return false;
//	}
//
//
//	e = cudaDeviceSynchronize();
//	if (e != cudaSuccess)
//	{
//		std::cout << "CUDA Sync Error: " << cudaGetErrorString(e) << "\n";
//		cudaFree(d_mask);
//		cudaFree(d_contourPoints);
//		cudaFree(d_contourValues);
//		cudaFree(d_output);
//		return false;
//	}
//
//
//	cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);
//
//	cudaFree(d_mask);
//	cudaFree(d_contourPoints);
//	cudaFree(d_contourValues);
//	cudaFree(d_output);
//	return true;
//}

cv::Mat maskedGaussianGPU(const cv::Mat& grayscale, const cv::Mat& mask, const cv::Mat& contour, float sigma)
{
	if (grayscale.empty() || mask.empty() || contour.empty()) 
	{
		std::cout << "Could not load images.\n";
		return cv::Mat();
	}

	if (grayscale.type() != CV_32F || mask.type() != CV_8UC1 || contour.type() != CV_8UC1) {
		std::cout << "Unexpected image types.\n";
		return cv::Mat();
	}

	// Extract contour points
	std::vector <cv::Point> contourPoints;
	std::vector<float> contourValues;
	for (int y = 0; y < contour.rows; ++y) {
		for (int x = 0; x < contour.cols; ++x) {
			if (contour.at<uchar>(y, x) > 0) {
				contourPoints.emplace_back(x, y);
				contourValues.emplace_back(grayscale.at<float>(x, y));
			}
		}
	}


	// computeWeightedAverage(mask.ptr<uchar>(0), &contourPoints[0], &contourValues[0], output.ptr<float>(0), grayscale.rows, grayscale.cols, contourPoints.size(), sigma);
	
	const size_t numContourPoints{ contourPoints.size() };

	size_t pointSize = numContourPoints * sizeof(cv::Point);
	size_t valueSize = numContourPoints * sizeof(float);


	cv::Point* d_contourPoints;
	float* d_contourValues;

	cudaMallocManaged(&d_contourPoints, pointSize);
	cudaMallocManaged(&d_contourValues, valueSize);
	cudaMemcpy(d_contourPoints, &contourPoints[0], pointSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_contourValues, &contourValues[0], valueSize, cudaMemcpyHostToDevice);

	const int rows{ grayscale.rows };
	const int cols{ grayscale.cols };

	dim3 blockSize(32, 32);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	// Ensure that gridSize.x and gridSize.y are within valid limits
	if (gridSize.x > 65535 || gridSize.y > 65535) {
		std::cout << "Grid size exceeds maximum limit." << std::endl;
		return cv::Mat();
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	if (blockSize.x * blockSize.y * blockSize.z > (size_t)prop.maxThreadsPerBlock)
	{
		cudaFree(d_contourPoints);
		cudaFree(d_contourValues);
		std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
		return cv::Mat();
	}
	
	cv::cuda::GpuMat maskGpu(mask);
	cv::cuda::GpuMat outputGpu(grayscale.size(), CV_32F);
	computeWeightedAverageKernel << <gridSize, blockSize >> > (maskGpu.ptr<uchar>(0), d_contourPoints, d_contourValues, outputGpu.ptr<float>(0),
		rows, cols, numContourPoints, sigma * sigma);

	cudaError_t e = cudaGetLastError();
	if (e != cudaSuccess)
	{
		cudaFree(d_contourPoints);
		cudaFree(d_contourValues);
		std::cout << "CUDA Launch Error: " << cudaGetErrorString(e) << "\n";
		return cv::Mat();
	}

	e = cudaDeviceSynchronize();
	if (e != cudaSuccess)
	{
		cudaFree(d_contourPoints);
		cudaFree(d_contourValues);
		std::cout << "CUDA Sync Error: " << cudaGetErrorString(e) << "\n";
		return cv::Mat();
	}

	cudaFree(d_contourPoints);
	cudaFree(d_contourValues);

	cv::Mat output;
	outputGpu.download(output);

	return output;
}