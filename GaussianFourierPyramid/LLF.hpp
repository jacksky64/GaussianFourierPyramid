#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <intrin.h>
#include <vector>
#include <algorithm>

#pragma region pyramidUtility
//image input version
static void buildLaplacianPyramid(const cv::Mat& src, std::vector<cv::Mat>& destPyramid, const int level)
{
	if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);

	cv::buildPyramid(src, destPyramid, level);
	for (int i = 0; i < level; i++)
	{
		cv::Mat temp;
		cv::pyrUp(destPyramid[i + 1], temp, destPyramid[i].size());
		cv::subtract(destPyramid[i], temp, destPyramid[i]);
	}
}
//pyramid input version
static void buildLaplacianPyramid(const std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& destPyramid, const int level)
{
	if (destPyramid.size() != level + 1) destPyramid.resize(level + 1);

	for (int l = 0; l < level; l++)
	{
		cv::Mat temp;
		cv::pyrUp(GaussianPyramid[l + 1], temp, GaussianPyramid[l].size());
		cv::subtract(GaussianPyramid[l], temp, destPyramid[l]);
	}
}

static void collapseLaplacianPyramid(const std::vector<cv::Mat>& LaplacianPyramid, cv::Mat& dest)
{
	const int level = (int)LaplacianPyramid.size();
	cv::Mat ret;
	cv::pyrUp(LaplacianPyramid[level - 1], ret, LaplacianPyramid[level - 2].size());
	for (int i = level - 2; i != 0; i--)
	{
		cv::add(ret, LaplacianPyramid[i], ret);
		cv::pyrUp(ret, ret, LaplacianPyramid[i - 1].size());
	}
	cv::add(ret, LaplacianPyramid[0], dest);
}


#pragma endregion

class MSEWeightedBlend
{
private:
	int get_simd_floor(const int val, const int simdwidth)
	{
		return (val / simdwidth) * simdwidth;
	}

	void wheightedCollapsePyramid(const std::vector<cv::Mat>& laplPyrMIP,
		const std::vector<cv::Mat>& gaussPyrMIP,
		const std::vector<cv::Mat>& laplPyrAvg,
		const std::vector<cv::Mat>& gaussPyrAvg,
		cv::Mat& dest)
	{
		const int level = (int)laplPyrAvg.size();
		std::vector<cv::Mat> restoredNorm;
		cv::Mat ret;

		cv::pyrUp(gaussPyrAvg[level - 1], ret, laplPyrAvg[level - 2].size());
		restoredNorm.push_back(ret.clone());
		for (int i = level - 2; i != 0; i--)
		{
			cv::add(ret, laplPyrAvg[i], ret);
			cv::pyrUp(ret, ret, laplPyrAvg[i - 1].size());
			restoredNorm.push_back(ret.clone());
		}
		cv::Mat dest0;
		cv::add(ret, laplPyrAvg[0], dest0);
		restoredNorm.push_back(dest0.clone());

		std::vector<double> power{ 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
		std::vector<double> alpha{ 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5 , 1.0, 0.2 };
		std::vector<double> beta { 1., 1., 1., 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

		cv::Mat vMask;
		cv::threshold(gaussPyrAvg[level - 1], vMask, 10., 1., cv::ThresholdTypes::THRESH_BINARY);

		//const double maxRange{ cv::mean(GaussianPyramid[level - 1], vMask>0)[0]};

		const double maxRange = 50;
		//std::vector<double> power{ 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 };
		//std::vector<double> alpha{ 1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 0.2, 0.2, 0.2 };
		//std::vector<double> beta{ 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

		std::vector<cv::Mat> restoredPost;

		// blend LPF from gaussian pyr
		cv::Mat blended;
		const double blendGaussAvg = 0.5;
		const double blendGaussMIP = 0;
		const double alphaBlendLapl = 0.3;

		cv::addWeighted(gaussPyrAvg[level - 1], blendGaussAvg, gaussPyrMIP[level - 1], blendGaussMIP, 0., blended);
		cv::pyrUp(blended, ret, laplPyrAvg[level - 2].size());

		restoredPost.push_back(ret.clone());

		for (int i = level - 2; i >= 0; i--)
		{
			cv::Mat gain{ gaussPyrAvg[i].clone() };
			cv::divide(gain, cv::Scalar(maxRange), gain);
			cv::pow(gain, power.at(i), gain);

			cv::addWeighted(laplPyrAvg[i], alphaBlendLapl, laplPyrMIP[i], 1. - alphaBlendLapl, 0., blended);

			cv::multiply(gain, blended, blended, beta.at(i));
			cv::multiply(ret, alpha.at(i), ret);
			cv::add(ret, blended, ret);

			if (i > 0)
				cv::pyrUp(ret, ret, laplPyrAvg[i - 1].size());

			restoredPost.push_back(ret.clone());
		}
		ret.copyTo(dest);
	}

public:
	void filter(const cv::Mat& srcLPF, const cv::Mat& srcHPF, const cv::Mat& mask, cv::Mat& dest, int level = 2)
	{
		CV_Assert(!srcLPF.empty() && !srcHPF.empty() && !mask.empty());

		CV_Assert((srcLPF.size() == srcHPF.size()) && (srcLPF.size() == mask.size()));

		if (dest.empty() || dest.size() != srcLPF.size() || dest.type() != srcLPF.type())
			dest.create(srcLPF.size(), srcLPF.type());

		std::vector<cv::Mat> gaussPyrLPF;    // level+1
		std::vector<cv::Mat> gaussPyrHPF;    // level+1
		std::vector<cv::Mat> gaussPyrMask; // level+1
		std::vector<cv::Mat> laplacianPyrLPF;   // level+1
		std::vector<cv::Mat> laplacianPyrHPF;   // level+1

		gaussPyrLPF.resize(level + 1);
		gaussPyrHPF.resize(level + 1);
		gaussPyrMask.resize(level + 1);

		if (srcLPF.depth() == CV_32F)
			srcLPF.copyTo(gaussPyrLPF[0]);
		else
			srcLPF.convertTo(gaussPyrLPF[0], CV_32F);

		if (srcHPF.depth() == CV_32F)
			srcHPF.copyTo(gaussPyrHPF[0]);
		else
			srcHPF.convertTo(gaussPyrHPF[0], CV_32F);

		if (mask.depth() == CV_32F)
			mask.copyTo(gaussPyrMask[0]);
		else
			mask.convertTo(gaussPyrMask[0], CV_32F);

		cv::normalize(gaussPyrMask[0], gaussPyrMask[0], 1, 0, cv::NormTypes::NORM_MINMAX, CV_32F);

		// Build Gaussian Pyramid
		cv::buildPyramid(gaussPyrLPF[0], gaussPyrLPF, level);
		cv::buildPyramid(gaussPyrHPF[0], gaussPyrHPF, level);
		cv::buildPyramid(gaussPyrMask[0], gaussPyrMask, level);

		// Build Laplacian Pyramids
		buildLaplacianPyramid(gaussPyrLPF, laplacianPyrLPF, level);
		buildLaplacianPyramid(gaussPyrHPF, laplacianPyrHPF, level);

		// Collapse Pyramids
		cv::Mat r;
		wheightedCollapsePyramid(laplacianPyrHPF, gaussPyrHPF, laplacianPyrLPF, gaussPyrLPF, r);

		r.convertTo(dest, srcLPF.type()); // convert 32F to output type
	}
};

#pragma region MSEblend
class MSEBlend
{
private:

	int get_simd_floor(const int val, const int simdwidth)
	{
		return (val / simdwidth) * simdwidth;
	}

	// Function to blend Laplacian pyramids
	std::vector<cv::Mat> blendPyramids(const std::vector<cv::Mat>& laplPyr1, const std::vector<cv::Mat>& laplPyr2, const std::vector<cv::Mat>& gaussPyrMask)
	{
		std::vector<cv::Mat> blendedPyr;
		for (size_t i = 0; i < laplPyr1.size(); ++i)
		{
			cv::Mat blended = laplPyr1[i].mul(gaussPyrMask[i]) + laplPyr2[i].mul(cv::Scalar::all(1.0) - gaussPyrMask[i]);
			blendedPyr.push_back(blended);
		}
		return blendedPyr;
	}

public:
	void filter(const cv::Mat& src1, const cv::Mat& src2, const cv::Mat& mask, cv::Mat& dest, int level = 2)
	{
		CV_Assert(!src1.empty() && !src2.empty() && !mask.empty());

		CV_Assert(src1.size() != src2.size() || src1.size() != mask.size());

		if (dest.empty() || dest.size() != src1.size() || dest.type() != src1.type())
			dest.create(src1.size(), src1.type());

		std::vector<cv::Mat> GaussianPyramid1;	//level+1
		std::vector<cv::Mat> GaussianPyramid2;	//level+1
		std::vector<cv::Mat> GaussianPyramidMask;	//level+1
		std::vector<cv::Mat> LaplacianPyramid1;	//level+1
		std::vector<cv::Mat> LaplacianPyramid2;	//level+1

		//(0) alloc
		GaussianPyramid1.resize(level + 1);
		GaussianPyramid2.resize(level + 1);
		GaussianPyramidMask.resize(level + 1);

		if (src1.depth() == CV_32F)
			src1.copyTo(GaussianPyramid1[0]);
		else
			src1.convertTo(GaussianPyramid1[0], CV_32F);

		if (src2.depth() == CV_32F)
			src2.copyTo(GaussianPyramid2[0]);
		else
			src2.convertTo(GaussianPyramid2[0], CV_32F);

		// Convert mask to float and normalize to [0, 1]
		//mask.convertTo(mask, CV_32F, 1.0 / 255.0);
		CV_Assert(mask.type() == CV_32F);

		//(1) Build Gaussian Pyramid
		cv::buildPyramid(GaussianPyramid1[0], GaussianPyramid1, level);
		cv::buildPyramid(GaussianPyramid2[0], GaussianPyramid2, level);
		cv::buildPyramid(GaussianPyramidMask[0], GaussianPyramidMask, level);

		//(2-2) Build Laplacian Pyramids
		buildLaplacianPyramid(GaussianPyramid1, LaplacianPyramid1, level);
		buildLaplacianPyramid(GaussianPyramid2, LaplacianPyramid2, level);

		//set last level
		LaplacianPyramid1[level] = GaussianPyramid1[level];
		LaplacianPyramid2[level] = GaussianPyramid2[level];

		// Blend Laplacian pyramids
		std::vector<cv::Mat> blendedPyr = blendPyramids(LaplacianPyramid1, LaplacianPyramid2, GaussianPyramidMask);

		//(4) Collapse Laplacian Pyramid to the last level
		collapseLaplacianPyramid(blendedPyr, blendedPyr[0]);

		blendedPyr[0].convertTo(dest, src1.type());//convert 32F to output type
	}
};
#pragma endregion



#pragma region MSEbilateral
class MSEBilateral
{
private:
	std::vector<cv::Mat> GaussianPyramid;////level+1
	std::vector<cv::Mat> LaplacianPyramid;//level+1

	int get_simd_floor(const int val, const int simdwidth)
	{
		return (val / simdwidth) * simdwidth;
	}

	void applyBilateral(const cv::Mat& src, cv::Mat& dest, const int filterSize, const float sigma_range, const float sigma_space)
	{
		cv::Mat r;
		// filter doesn't work inplace
		cv::bilateralFilter(src, r, filterSize, sigma_range, sigma_space);
		r.copyTo(dest);
	}

public:
	void filter(const cv::Mat& src, cv::Mat& dest, std::vector<float> sigma_range, float sigma_space, int filterSize, int level = 2)
	{
		CV_Assert(!sigma_range.empty());

		//main processing 
		dest.create(src.size(), src.type());
		//(0) alloc
		if (GaussianPyramid.size() != level + 1)GaussianPyramid.resize(level + 1);

		if (src.depth() == CV_32F) src.copyTo(GaussianPyramid[0]);
		else src.convertTo(GaussianPyramid[0], CV_32F);

		//(1) Build Gaussian Pyramid
		cv::buildPyramid(GaussianPyramid[0], GaussianPyramid, level);

		//(2-2) Build Laplacian Pyramids
		buildLaplacianPyramid(GaussianPyramid, LaplacianPyramid, level);

		//(2) remap Laplacian Pyramids
		for (int n = 0; n < level; n++)
		{
			applyBilateral(LaplacianPyramid[n], LaplacianPyramid[n], filterSize, sigma_range.at(std::min(size_t(n), sigma_range.size() - 1)), sigma_space);
		}

		//set last level
		LaplacianPyramid[level] = GaussianPyramid[level];

		//(4) Collapse Laplacian Pyramid to the last level
		collapseLaplacianPyramid(LaplacianPyramid, LaplacianPyramid[0]);

		LaplacianPyramid[0].convertTo(dest, src.type());//convert 32F to output type
	}

	void filterHPF(const cv::Mat& src, cv::Mat& destHF, float sigma_range, float sigma_space, int filterSize, int level = 2)
	{
		//main processing 
		destHF.create(src.size(), src.type());
		//(0) alloc
		if (GaussianPyramid.size() != level + 1)GaussianPyramid.resize(level + 1);

		if (src.depth() == CV_32F) src.copyTo(GaussianPyramid[0]);
		else src.convertTo(GaussianPyramid[0], CV_32F);

		//(1) Build Gaussian Pyramid
		cv::buildPyramid(GaussianPyramid[0], GaussianPyramid, level);

		//(2-2) Build Laplacian Pyramids
		buildLaplacianPyramid(GaussianPyramid, LaplacianPyramid, level);

		//(2) remap Laplacian Pyramids
		for (int n = 0; n < level; n++)
		{
			applyBilateral(LaplacianPyramid[n], LaplacianPyramid[n], filterSize, sigma_range, sigma_space);
		}

		//set last level
		LaplacianPyramid[level].create(GaussianPyramid[level].size(), GaussianPyramid[level].type());
		LaplacianPyramid[level].setTo(0.);

		//(4) Collapse Laplacian Pyramid to the last level
		collapseLaplacianPyramid(LaplacianPyramid, LaplacianPyramid[0]);

		LaplacianPyramid[0].convertTo(destHF, src.type());//convert 32F to output type
	}
};
#pragma endregion


#pragma region MSE
class MSEGaussRemap
{
private:
	std::vector<cv::Mat> GaussianPyramid;////level+1
	std::vector<cv::Mat> LaplacianPyramid;//level+1

	int get_simd_floor(const int val, const int simdwidth)
	{
		return (val / simdwidth) * simdwidth;
	}

	void remap(const cv::Mat& src, cv::Mat& dest, const float g, const float sigma_range, const float boost)
	{
		if (src.data != dest.data) dest.create(src.size(), CV_32F);
		if (1)
		{
			const float* s = src.ptr<float>();
			float* d = dest.ptr<float>();
			const int size = src.size().area();
			const int SIZE32 = get_simd_floor(size, 32);
			const int SIZE8 = get_simd_floor(size, 8);
			const __m256 mg = _mm256_set1_ps(g);
			const float coeff = float(1.0 / (-2.0 * sigma_range * sigma_range));
			const __m256 mcoeff = _mm256_set1_ps(coeff);
			const __m256 mdetail = _mm256_set1_ps(boost);

			__m256 ms, subsg;
			for (int i = 0; i < SIZE32; i += 32)
			{
				ms = _mm256_loadu_ps(s + i);
				subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

				ms = _mm256_loadu_ps(s + i + 8);
				subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i + 8, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

				ms = _mm256_loadu_ps(s + i + 16);
				subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i + 16, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));

				ms = _mm256_loadu_ps(s + i + 24);
				subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i + 24, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
			}
			for (int i = SIZE32; i < SIZE8; i += 8)
			{
				ms = _mm256_loadu_ps(s + i);
				subsg = _mm256_sub_ps(ms, mg);
				_mm256_storeu_ps(d + i, _mm256_fmadd_ps(subsg, _mm256_mul_ps(mdetail, _mm256_exp_ps(_mm256_mul_ps(_mm256_mul_ps(subsg, subsg), mcoeff))), ms));
			}

			for (int i = SIZE8; i < size; i++)
			{
				const float x = s[i] - g;
				d[i] = x * boost * exp(x * x * coeff) + s[i];
			}
		}
		else
		{
			// debug
			const int size = src.size().area();
			const float* s = src.ptr<float>();
			float* d = dest.ptr<float>();
			const float coeff = float(1.0 / (-2.0 * sigma_range * sigma_range));

			for (int i = 0; i < size; i++)
			{
				const float x = s[i] - g;
				d[i] = x * boost * exp(x * x * coeff) + s[i];
			}
		}
	}

public:
	void filter(const cv::Mat& src, cv::Mat& dest, float sigma_range, const std::vector<float>& boost, int level = 2)
	{
		dest.create(src.size(), src.type());
		//(0) alloc
		if (GaussianPyramid.size() != level + 1)GaussianPyramid.resize(level + 1);

		if (src.depth() == CV_32F) src.copyTo(GaussianPyramid[0]);
		else src.convertTo(GaussianPyramid[0], CV_32F);

		//(1) Build Gaussian Pyramid
		cv::buildPyramid(GaussianPyramid[0], GaussianPyramid, level);

		//(2-2) Build Laplacian Pyramids
		buildLaplacianPyramid(GaussianPyramid, LaplacianPyramid, level);

		//(2) remap Laplacian Pyramids
		for (int n = 0; n < level; n++)
		{
			const float levelBoost = (n < boost.size() ? boost.at(n) : boost.at(boost.size() - 1));
			remap(LaplacianPyramid[n], LaplacianPyramid[n], 0.f, sigma_range, levelBoost);
		}

		//set last level
		LaplacianPyramid[level].create(GaussianPyramid[level].size(), GaussianPyramid[level].type());
		//LaplacianPyramid[level].setTo(0.f);
		LaplacianPyramid[level] = GaussianPyramid[level];

		//(4) Collapse Laplacian Pyramid to the last level
		collapseLaplacianPyramid(LaplacianPyramid, LaplacianPyramid[0]);

		LaplacianPyramid[0].convertTo(dest, src.type());//convert 32F to output type
	}
};

#pragma endregion


#pragma region LLF
// Fast Local Laplacian Filter
// M. Aubry, S. Paris, J. Kautz, and F. Durand, "Fast local laplacian filters: Theory and applications," ACM Transactionson Graphics, vol. 33, no. 5, 2014.
// Z. Qtang, L. He, Y. Chen, X. Chen, and D. Xu, "Adaptive fast local laplacian filtersand its edge - aware application," Multimedia Toolsand Applications, vol. 78, pp. 5, 2019.
class FastLLF
{
private:
	const float intensityMin = 0.f;
	const float intensityMax = 255.f;
	const float intensityRange = 255.f;
	const int rangeMax = 256;

	std::vector<cv::Mat> GaussianPyramid;////level+1
	std::vector<std::vector<cv::Mat>> LaplacianPyramidOrder;//(level+1) x order
	std::vector<cv::Mat> LaplacianPyramid;//level+1

	bool isAdaptive = false;
	float sigma_range = 0.f; //\sigma_r in Eq. (6)
	float boost = 1.f;//m in Eq. (6)
	cv::Mat adaptiveSigmaMap;//Sec III.B Pixel-by-pixel enhancement
	cv::Mat adaptiveBoostMap;//Sec III.B Pixel-by-pixel enhancement

	//compute interval parameter in linear interpolation (Sec. II.D. Fast Local Laplacian Filtering)
	float getTau(const int k, const int order)
	{
		const float delta = intensityRange / (order - 1);
		return float(k * delta + intensityMin);
	}

	void remap(const cv::Mat& src, cv::Mat& dest, const float g, const float sigma_range, const float boost)
	{
		if (src.data != dest.data) dest.create(src.size(), CV_32F);

		const int size = src.size().area();
		const float* s = src.ptr<float>();
		float* d = dest.ptr<float>();
		const float coeff = float(1.0 / (-2.0 * sigma_range * sigma_range));

		for (int i = 0; i < size; i++)
		{
			const float x = s[i] - g;
			d[i] = x * boost * exp(x * x * coeff) + s[i];
		}
	}
	void remapAdaptive(const cv::Mat& src, cv::Mat& dest, const float g, const cv::Mat& sigma_range, const cv::Mat& boost)
	{
		if (src.data != dest.data) dest.create(src.size(), CV_32F);

		const int size = src.size().area();
		const float* s = src.ptr<float>();
		float* d = dest.ptr<float>();
		const float* sigmaptr = sigma_range.ptr<float>();
		const float* boostptr = boost.ptr<float>();

		for (int i = 0; i < size; i++)
		{
			const float coeff = 1.f / (-2.f * sigmaptr[i] * sigmaptr[i]);
			const float boost = boostptr[i];
			const float x = s[i] - g;
			d[i] = x * boost * exp(x * x * coeff) + s[i];
		}
	}

	inline void getLinearIndex(float v, int& index_l, int& index_h, float& alpha, const int order, const float intensityMin, const float intensityMax)
	{
		const float intensityRange = intensityMax - intensityMin;
		const float delta = intensityRange / (order - 1);
		const int i = (int)(v / delta);

		if (i < 0)
		{
			index_l = 0;
			index_h = 0;
			alpha = 1.f;
		}
		else if (i + 1 > order - 1)
		{
			index_l = order - 1;
			index_h = order - 1;
			alpha = 0.f;
		}
		else
		{
			index_l = i;
			index_h = i + 1;
			alpha = 1.f - (v - (i * delta)) / (delta);
		}
	}
	//last level is not blended; thus, inplace operation for input Gaussian Pyramid is required.
	void blendLaplacianLinear(const std::vector<std::vector<cv::Mat>>& LaplacianPyramid, std::vector<cv::Mat>& GaussianPyramid, std::vector<cv::Mat>& destPyramid, const int order)
	{
		const int level = (int)GaussianPyramid.size();
		destPyramid.resize(level);
		std::vector<const float*> lptr(order);
		for (int l = 0; l < level - 1; l++)
		{
			destPyramid[l].create(GaussianPyramid[l].size(), CV_32F);
			float* g = GaussianPyramid[l].ptr<float>();
			float* d = destPyramid[l].ptr<float>();
			for (int k = 0; k < order; k++)
			{
				lptr[k] = LaplacianPyramid[k][l].ptr<float>();
			}

			for (int i = 0; i < GaussianPyramid[l].size().area(); i++)
			{
				float alpha;
				int high, low;
				getLinearIndex(g[i], low, high, alpha, order, intensityMin, intensityMax);
				d[i] = alpha * lptr[low][i] + (1.f - alpha) * lptr[high][i];
			}
		}
	}

	void setAdaptive(const cv::Mat& sigmaMap, const cv::Mat& boostMap)
	{
		isAdaptive = true;
		this->adaptiveSigmaMap = sigmaMap;
		this->adaptiveBoostMap = boostMap;
	}
	void setFix(const float sigma_range, const float boost)
	{
		isAdaptive = false;
		this->sigma_range = sigma_range;
		this->boost = boost;
	}

	//grayscale processing
	void gray(const cv::Mat& src, cv::Mat& dest, const int order, const int level)
	{
		//(0) alloc
		if (GaussianPyramid.size() != level + 1)GaussianPyramid.resize(level + 1);
		LaplacianPyramidOrder.resize(order);
		for (int n = 0; n < order; n++)
		{
			LaplacianPyramidOrder[n].resize(level + 1);
		}
		if (src.depth() == CV_32F) src.copyTo(GaussianPyramid[0]);
		else src.convertTo(GaussianPyramid[0], CV_32F);

		//(1) Build Gaussian Pyramid
		cv::buildPyramid(GaussianPyramid[0], GaussianPyramid, level);

		//(2) Build remapped Laplacian Pyramids
		for (int n = 0; n < order; n++)
		{
			//(2-1) Remap Input Image
			if (isAdaptive)
			{
				remapAdaptive(GaussianPyramid[0], LaplacianPyramidOrder[n][0], getTau(n, order), adaptiveSigmaMap, adaptiveBoostMap);
			}
			else
			{
				remap(GaussianPyramid[0], LaplacianPyramidOrder[n][0], getTau(n, order), sigma_range, boost);
			}

			//(2-2) Build Remapped Laplacian Pyramids
			buildLaplacianPyramid(LaplacianPyramidOrder[n][0], LaplacianPyramidOrder[n], level);
		}

		//(3) interpolate Laplacian Pyramid from Remapped Laplacian Pyramids
		blendLaplacianLinear(LaplacianPyramidOrder, GaussianPyramid, LaplacianPyramid, order);
		//set last level
		LaplacianPyramid[level] = GaussianPyramid[level];

		//(4) Collapse Laplacian Pyramid to the last level
		collapseLaplacianPyramid(LaplacianPyramid, LaplacianPyramid[0]);

		LaplacianPyramid[0].convertTo(dest, src.depth());//convert 32F to output type
	}
	//main processing (same methods: Fast LLF and Fourier LLF)
	void body(const cv::Mat& src, cv::Mat& dest, const int order, const int level)
	{
		dest.create(src.size(), src.type());
		if (src.channels() == 1)
		{
			gray(src, dest, order, level);
		}
		else
		{
			const bool onlyY = true;
			if (onlyY)
			{
				cv::Mat gim;
				cv::cvtColor(src, gim, cv::COLOR_BGR2YUV);
				std::vector<cv::Mat> vsrc;
				cv::split(gim, vsrc);
				gray(vsrc[0], vsrc[0], order, level);
				merge(vsrc, dest);
				cv::cvtColor(dest, dest, cv::COLOR_YUV2BGR);
			}
			else
			{
				std::vector<cv::Mat> vsrc;
				std::vector<cv::Mat> vdst(3);
				cv::split(src, vsrc);
				gray(vsrc[0], vdst[0], order, level);
				gray(vsrc[1], vdst[1], order, level);
				gray(vsrc[2], vdst[2], order, level);
				cv::merge(vdst, dest);
			}
		}
	}
public:
	//fix parameter (sigma_range and boost)
	void filter(const cv::Mat& src, cv::Mat& dest, const int order, const float sigma_range, const float boost, const int level = 2)
	{
		setFix(sigma_range, boost);
		body(src, dest, order, level);
	}
	//adaptive parameter (sigma_range and boost)
	void filter(const cv::Mat& src, cv::Mat& dest, const int order, const cv::Mat& sigma_range, const cv::Mat& boost, const int level = 2)
	{
		setAdaptive(sigma_range, boost);
		body(src, dest, order, level);
	}
};

// Fourier Local Laplacian Filter
// Y. Sumiya, T. Otsuka, Y. Maedaand N. Fukushima, "Gaussian Fourier Pyramid for Local Laplacian Filter," IEEE Signal Processing Letters, vol. 29, pp. 11-15, 2022.
class GaussianFourierLLF
{
private:
	const float intensityMin = 0.f;
	const float intensityMax = 255.f;
	const float intensityRange = 255.f;
	const int rangeMax = 256;

	float T = 0.f;
	std::vector<float> alpha, beta;
	std::vector<float> omega;//(CV_2PI*(k + 1)/T)

	std::vector<cv::Mat> FourierPyramidSin; //level+1
	std::vector<cv::Mat> FourierPyramidCos; //level+1
	std::vector<cv::Mat> LaplacianPyramid; //level+1
	std::vector<cv::Mat> GaussianPyramid; //level+1

	bool isAdaptive = false;
	float sigma_range = 0.f;//\sigma_r in Eq. (6)
	float boost = 1.f;//m in Eq. (6)
	int level = 0;
	std::vector<cv::Mat> adaptiveSigmaMap;//Sec III.B Pixel-by-pixel enhancement
	std::vector<cv::Mat> adaptiveBoostMap;//Sec III.B Pixel-by-pixel enhancement

	double df(double x, const int K, const double Irange, const double sigma_range)
	{
		const double s = sigma_range / Irange;
		const double kappa = (2 * K + 1) * CV_PI;
		const double psi = kappa * s / x;
		const double phi = (x - 1.0) / s;
		return (-kappa * exp(-phi * phi) + psi * psi * exp(-psi * psi));
	}
	double computeT_ClosedForm(int order, double sigma_range, const double intensityRange)
	{
		double x, diff;

		double x1 = 1.0, x2 = 15.0;
		int loop = 20;
		for (int i = 0; i < loop; ++i)
		{
			x = (x1 + x2) / 2.0;
			diff = df(x, order, intensityRange, sigma_range);
			((0.0 <= diff) ? x2 : x1) = x;
		}
		return x;
	}
	void initRangeFourier(const int order, const float sigma_range, const float boost)
	{
		if (alpha.size() != order)
		{
			alpha.resize(order);
			beta.resize(order);
		}

		if (omega.size() != order) omega.resize(order);

		T = float(intensityRange * computeT_ClosedForm(order, sigma_range, intensityRange));//Eq. (12), detail information is in K. Sugimoto and S. Kamata, "Compressive bilateral filtering," IEEE Transactions on Image Processing, vol. 24, no. 11, pp.3357-3369, 2015.

		//compute omega and alpha in Eqs. (9) and (10)
		for (int k = 0; k < order; k++)
		{
			omega[k] = float(CV_2PI / (double)T * (double)(k + 1));
			const double coeff_kT = omega[k] * sigma_range;
			alpha[k] = float(2.0 * exp(-0.5 * coeff_kT * coeff_kT) * sqrt(CV_2PI) * sigma_range / T);
		}
	}

	void remapCos(const cv::Mat& src, cv::Mat& dest, const float omega)
	{
		dest.create(src.size(), CV_32F);
		const float* s = src.ptr<float>();
		float* d = dest.ptr<float>();
		const int size = src.size().area();

		for (int i = 0; i < size; i++)
		{
			d[i] = cos(omega * s[i]);
		}
	}
	void remapSin(const cv::Mat& src, cv::Mat& dest, const float omega)
	{
		dest.create(src.size(), CV_32F);
		const float* s = src.ptr<float>();
		float* d = dest.ptr<float>();
		const int size = src.size().area();

		for (int i = 0; i < size; i++)
		{
			d[i] = sin(omega * s[i]);
		}
	}

	void productSumPyramidLayer(const cv::Mat& srccos, const cv::Mat& srcsin, const cv::Mat gauss, cv::Mat& dest, const float omega, const float alpha, const float sigma, const float boost)
	{
		dest.create(srccos.size(), CV_32F);

		const int size = srccos.size().area();

		const float* c = srccos.ptr<float>();
		const float* s = srcsin.ptr<float>();
		const float* g = gauss.ptr<float>();
		float* d = dest.ptr<float>();

		const float lalpha = -sigma * sigma * omega * alpha * boost;

		for (int i = 0; i < size; i++)
		{
			const float ms = omega * g[i];
			d[i] += lalpha * (sin(ms) * c[i] - cos(ms) * (s[i]));
		}
	}
	inline float getAdaptiveAlpha(float coeff, float base, float sigma, float boost)
	{
		const float a = coeff * sigma;
		return sigma * sigma * sigma * boost * base * exp(-0.5f * a * a);
	}
	void productSumAdaptivePyramidLayer(const cv::Mat& srccos, const cv::Mat& srcsin, const cv::Mat gauss, cv::Mat& dest, const float omega, const float alpha, const cv::Mat& sigma, const cv::Mat& boost)
	{
		dest.create(srccos.size(), CV_32F);

		const int size = srccos.size().area();

		const float* c = srccos.ptr<float>();
		const float* s = srcsin.ptr<float>();
		const float* g = gauss.ptr<float>();
		float* d = dest.ptr<float>();

		const float base = -float(2.0 * sqrt(CV_2PI) * omega / T);
		const float* adaptiveSigma = sigma.ptr<float>();
		const float* adaptiveBoost = boost.ptr<float>();

		for (int i = 0; i < size; i++)
		{
			const float lalpha = getAdaptiveAlpha(omega, base, adaptiveSigma[i], adaptiveBoost[i]);
			const float ms = omega * g[i];
			d[i] += lalpha * (sin(ms) * c[i] - cos(ms) * (s[i]));
		}
	}

	void setAdaptive(const cv::Mat& sigmaMap, const cv::Mat& boostMap, const int level)
	{
		isAdaptive = true;
		cv::buildPyramid(sigmaMap, adaptiveSigmaMap, level);
		cv::buildPyramid(boostMap, adaptiveBoostMap, level);
	}
	void setFix(const float sigma_range, const float boost)
	{
		isAdaptive = false;
		this->sigma_range = sigma_range;
		this->boost = boost;
	}

	//grayscale processing
	void gray(const cv::Mat& src, cv::Mat& dest, const int order, const int level)
	{
		//(0) alloc
		GaussianPyramid.resize(level + 1);
		FourierPyramidCos.resize(level + 1);
		FourierPyramidSin.resize(level + 1);
		if (src.depth() == CV_8U) src.convertTo(GaussianPyramid[0], CV_32F);
		else src.copyTo(GaussianPyramid[0]);

		//compute alpha, omega, T
		initRangeFourier(order, sigma_range, boost);


		//(1) Build Gaussian Pyramid
		cv::buildPyramid(GaussianPyramid[0], GaussianPyramid, level);

		//(2) Build Laplacian Pyramid
		//(2-1) Build Laplacian Pyramid for DC
		buildLaplacianPyramid(GaussianPyramid, LaplacianPyramid, level);

		for (int k = 0; k < order; k++)
		{
			// (2-2) Build Remapped Laplacian Pyramid for Cos
			remapCos(GaussianPyramid[0], FourierPyramidCos[0], omega[k]);
			//build cos Gaussian pyramid and then generate Laplacian pyramid
			buildLaplacianPyramid(FourierPyramidCos[0], FourierPyramidCos, level);
			// (2-3) Build Remapped Laplacian Pyramid for Sin
			remapSin(GaussianPyramid[0], FourierPyramidSin[0], omega[k]);
			//build sin Gaussian pyramid and then generate Laplacian pyramid
			buildLaplacianPyramid(FourierPyramidSin[0], FourierPyramidSin, level);

			// (3) product-sum Gaussian Fourier pyramid
			if (isAdaptive)
			{
				for (int l = 0; l < level; l++)
				{
					productSumAdaptivePyramidLayer(FourierPyramidCos[l], FourierPyramidSin[l], GaussianPyramid[l], LaplacianPyramid[l], omega[k], alpha[k], adaptiveSigmaMap[l], adaptiveBoostMap[l]);
				}
			}
			else
			{
				for (int l = 0; l < level; l++)
				{
					productSumPyramidLayer(FourierPyramidCos[l], FourierPyramidSin[l], GaussianPyramid[l], LaplacianPyramid[l], omega[k], alpha[k], sigma_range, boost);
				}
			}
		}
		//set last level
		LaplacianPyramid[level] = GaussianPyramid[level];

		//(4) Collapse Laplacian Pyramid
		collapseLaplacianPyramid(LaplacianPyramid, LaplacianPyramid[0]);

		LaplacianPyramid[0].convertTo(dest, src.depth());//convert 32F to output type
	}
	//main processing (same methods: Fast LLF and Fourier LLF)
	void body(const cv::Mat& src, cv::Mat& dest, const int order, const int level)
	{
		dest.create(src.size(), src.type());
		if (src.channels() == 1)
		{
			gray(src, dest, order, level);
		}
		else
		{
			const bool onlyY = true;
			if (onlyY)
			{
				cv::Mat gim;
				cv::cvtColor(src, gim, cv::COLOR_BGR2YUV);
				std::vector<cv::Mat> vsrc;
				cv::split(gim, vsrc);
				gray(vsrc[0], vsrc[0], order, level);
				merge(vsrc, dest);
				cv::cvtColor(dest, dest, cv::COLOR_YUV2BGR);
			}
			else
			{
				std::vector<cv::Mat> vsrc;
				std::vector<cv::Mat> vdst(3);
				cv::split(src, vsrc);
				gray(vsrc[0], vdst[0], order, level);
				gray(vsrc[1], vdst[1], order, level);
				gray(vsrc[2], vdst[2], order, level);
				cv::merge(vdst, dest);
			}
		}
	}
public:
	//fix parameter (sigma_range and boost: same methods: Fast LLF and Fourier LLF)
	void filter(const cv::Mat& src, cv::Mat& dest, const int order, const float sigma_range, const float boost, const int level = 2)
	{
		setFix(sigma_range, boost);
		body(src, dest, order, level);
	}
	//adaptive parameter (sigma_range and boost: same methods: Fast LLF and Fourier LLF)
	void filter(const cv::Mat& src, cv::Mat& dest, const int order, const cv::Mat& sigma_range, const cv::Mat& boost, const int level = 2)
	{
		setAdaptive(sigma_range, boost, level);
		body(src, dest, order, level);
	}
};
#pragma endregion


#pragma region gammaEnh
class gammaEnh
{
public:
	void applyAdaptiveGamma(cv::Mat& srcDest, const cv::Mat& mask)
	{
		CV_Assert(!srcDest.empty() && srcDest.type() == CV_16U);

		std::vector<cv::Mat> s{ srcDest };
		std::vector<cv::Mat> m{ mask };

		std::vector<float> normCumHist;
		calcCumHist(s, m, normCumHist);

		// Compute the transformation function
		cv::Mat lut = buildAdaptiveGammaLutCDF(normCumHist);

		// Apply the lookup table
		remap(lut, s);
	}

	void applyAdaptiveGamma(std::vector<cv::Mat>& s, const std::vector<cv::Mat>& m)
	{
		std::vector<float> normCumHist;
		calcCumHist(s, m, normCumHist);

		// Compute the transformation function
		cv::Mat lut = buildAdaptiveGammaLutCDF(normCumHist);

		// Apply the lookup table
		remap(lut, s);
	}

	void applyAdaptiveGammaCDF(cv::Mat& srcDest, const cv::Mat& mask)
	{
		CV_Assert(!srcDest.empty() && srcDest.type() == CV_16U);

		std::vector<cv::Mat> s{ srcDest };
		std::vector<cv::Mat> m{ mask };

		std::vector<float> normCumHist;
		calcCumHistCDF(s, m, normCumHist);

		// Compute the transformation function
		cv::Mat lut = buildAdaptiveGammaLutCDF(normCumHist);

		// Apply the lookup table
		remap(lut, s);
	}

	void applyAdaptiveGammaCDF(std::vector<cv::Mat>& s, const std::vector<cv::Mat>& m)
	{
		std::vector<float> normCumHist;
		calcCumHistCDF(s, m, normCumHist);

		// Compute the transformation function
		cv::Mat lut = buildAdaptiveGammaLutCDF(normCumHist);

		// Apply the lookup table
		remap(lut, s);
	}

	void applyGamma(cv::Mat& srcDest, const cv::Mat& mask, double gamma, double tailPercent = 0.001, double enlargeRangePerc = 0.1)
	{
		ushort minValue{ 0 };
		ushort maxValue{ 0 };

		std::vector<cv::Mat> s{ srcDest };
		std::vector<cv::Mat> m{ mask };

		findMinMax(s, m, minValue, maxValue, tailPercent);

		// enlarge range
		const double range{ double(maxValue) - double(minValue) };
		const double rangeAdj{ range * enlargeRangePerc };

		const double enlargedMinValue = std::max(0., double(minValue) - rangeAdj);
		const double enlargedMaxValue = std::min(65535., double(maxValue) + rangeAdj);
		const double enlargedRange{ enlargedMaxValue - enlargedMinValue };

		gammaCorrection(s, gamma, ushort(enlargedMinValue), ushort(enlargedMaxValue));
	}

	void applyGamma(std::vector<cv::Mat>& srcDest, const std::vector<cv::Mat>& mask, double gamma, double tailPercent = 0.001, double enlargeRangePerc = 0.1)
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
	cv::Mat buildAdaptiveGammaLutCDF(const std::vector<float>& normCumHist)
	{
		// Determine total number of pixels
		int totalPixels = normCumHist[normCumHist.size() - 1];

		// Determine the cutoff values
		const double tailPercent = 0.0001;
		float upperCutoff = float(double(totalPixels) * (1. - tailPercent));
		float lowerCutoff = int(double(totalPixels) * tailPercent);

		int maxValue{ 0 };
		// Find maxValue
		for (int i = normCumHist.size() - 1; i >= 0; --i) {
			if (normCumHist[i] < upperCutoff) {
				maxValue = i;
				break;
			}
		}

		// Find minValue
		int minValue{ 0 };
		for (int i = 0; i < normCumHist.size() - 1; ++i) {
			if (normCumHist[i] > lowerCutoff) {
				minValue = i;
				break;
			}
		}


		// Create a look-up table
		cv::Mat lut(1, 65536, CV_16UC1);
		ushort* lut_data = lut.ptr<ushort>();

		for (int i = 0; i < normCumHist.size(); ++i)
		{
			int iVal{ i };
			if (iVal < minValue)
				iVal = minValue;

			if (iVal > maxValue)
				iVal = maxValue;
			lut_data[i] = cv::saturate_cast<ushort>(pow((float(iVal) - float(minValue)) / (float(maxValue) - float(minValue)), 1. - normCumHist[i]) * float(normCumHist.size() - 1));
		}

		return lut;
	}
	cv::Mat buildAdaptiveGammaLut(double gamma, const std::vector<float>& normCumHist)
	{
		CV_Assert(gamma >= 0);

		// Compute the transformation function
		std::vector<ushort> lookupTable(normCumHist.size());
		for (int i = 0; i < normCumHist.size(); ++i)
			lookupTable[i] = cv::saturate_cast<ushort>(pow(normCumHist[i], gamma) * 65535.);


		// Create a look-up table
		cv::Mat lut(1, 65536, CV_16UC1);
		ushort* lut_data = lut.ptr<ushort>();

		for (int i = 0; i < 65536; ++i)
			lut_data[i] = cv::saturate_cast<ushort>(pow(double(normCumHist[i]), gamma) * 65535.);

		return lut;
	}

	void remap(const cv::Mat& lut, std::vector<cv::Mat>& srcDest)
	{
		// Apply the lookup table
		const ushort* lut_data = lut.ptr<ushort>();
		for (int nImage = 0; nImage < srcDest.size(); nImage++)
		{
			cv::Mat s(srcDest[nImage]);

			CV_Assert(!s.empty());
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

	void calcCumHist(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& mask, std::vector<float>& cumulative)
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
			auto s{ src.at(n) };
			auto m{ mask.at(n) };
			CV_Assert(s.type() == CV_16U);
			CV_Assert(m.type() == CV_8U);
			cv::calcHist(&s, 1, 0, m, hist, 1, &histSize, &histRange, uniform, accumulate);
		}

		// Calculate the cumulative histogram
		cumulative.resize(histSize);
		cumulative[0] = (float)hist.at<float>(0);

		for (int i = 1; i < cumulative.size(); ++i)
			cumulative[i] = cumulative[i - 1] + (float)hist.at<float>(i);

		// Normalize the cumulative histogram
		float totalPixels = cumulative[cumulative.size() - 1];

		if (totalPixels > 0.f)
		{
			for (int i = 0; i < cumulative.size(); ++i)
				cumulative[i] = cumulative[i] / totalPixels;
		}
	}


	void calcCumHistCDF(const std::vector<cv::Mat>& src, const std::vector<cv::Mat>& mask, std::vector<float>& cumulative)
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
			auto s{ src.at(n) };
			auto m{ mask.at(n) };
			CV_Assert(s.type() == CV_16U);
			CV_Assert(m.type() == CV_8U);
			cv::calcHist(&s, 1, 0, m, hist, 1, &histSize, &histRange, uniform, accumulate);
		}

		// Calculate the cumulative histogram
		cumulative.resize(histSize);
		cumulative[0] = (float)hist.at<float>(0);

		for (int i = 1; i < cumulative.size(); ++i)
			cumulative[i] = cumulative[i - 1] + (float)hist.at<float>(i);

		float totalPixels = cumulative[cumulative.size() - 1];
		if (totalPixels > 0.f)
		{
			for (int i = 0; i < cumulative.size(); ++i)
				cumulative[i] = cumulative[i] / totalPixels;
		}

		// modify pdf
		float pdfMin = 0.f;
		float pdfMax = 1.f;

		// Find minValue
		for (int i = 0; i < histSize; ++i)
		{
			if ((float)hist.at<float>(i) < pdfMin)
				pdfMin = (float)hist.at<float>(i);
		}

		// Find maxValue
		for (int i = 0; i < histSize; ++i)
		{
			if ((float)hist.at<float>(i) > pdfMax)
				pdfMax = (float)hist.at<float>(i);
		}

		// modify pdf
		std::vector<float> pdfw(histSize);
		for (int i = 1; i < histSize; ++i)
		{
			const float p = pdfMax * powf(((float)hist.at<float>(i) - pdfMin) / (pdfMax - pdfMin), cumulative[i]);
			pdfw[i] = p;
		}

		// Calculate the cumulative histogram
		cumulative.resize(histSize);
		cumulative[0] = pdfw[0];

		for (int i = 1; i < cumulative.size(); ++i)
			cumulative[i] = cumulative[i - 1] + pdfw[i];

		// Normalize the cumulative histogram
		float tv = float(cumulative[cumulative.size() - 1]);

		if (tv > 0.f)
		{
			for (int i = 0; i < cumulative.size(); ++i)
				cumulative[i] = cumulative[i] / tv;
		}
	}


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
			auto s{ src.at(n) };
			auto m{ mask.at(n) };
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
	void gammaCorrection(std::vector<cv::Mat>& src, double gamma, ushort minValue, ushort maxValue)
	{
		cv::Mat lut = buildGammaLut(gamma, minValue, maxValue);

		remap(lut, src);
	}

	cv::Mat buildGammaLut(double gamma, ushort minValue, ushort maxValue)
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


#pragma region sigmoidEnh
class sigmoidEnh
{
public:
	void apply(const cv::Mat& srcDest, const cv::Mat& mask, double sigma, double tailPercent = 0.001, double enlargeRangePerc = 0.1)
	{
		ushort minValue{ 0 };
		ushort maxValue{ 0 };

		std::vector<cv::Mat> s{ srcDest };
		std::vector<cv::Mat> m{ mask };

		findMinMax(s, m, minValue, maxValue, tailPercent);

		// enlarge range
		const double range{ double(maxValue) - double(minValue) };
		const double rangeAdj{ range * enlargeRangePerc };

		const double enlargedMinValue = std::max(0., double(minValue) - rangeAdj);
		const double enlargedMaxValue = std::min(65535., double(maxValue) + rangeAdj);
		const double enlargedRange{ enlargedMaxValue - enlargedMinValue };

		doLut(s, sigma, ushort(enlargedMinValue), ushort(enlargedMaxValue));
	}

	void apply(const std::vector<cv::Mat>& srcDest, const std::vector<cv::Mat>& mask, double sigma, double tailPercent = 0.001, double enlargeRangePerc = 0.1)
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

		doLut(srcDest, sigma, ushort(enlargedMinValue), ushort(enlargedMaxValue));
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
			auto s{ src.at(n) };
			auto m{ mask.at(n) };
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
	void doLut(const std::vector<cv::Mat>& src, double sigma, ushort minValue, ushort maxValue)
	{
		cv::Mat lut = lutBuild(sigma, minValue, maxValue);
		const ushort* lut_data = lut.ptr<ushort>();

		for (int nImage = 0; nImage < src.size(); nImage++)
		{
			cv::Mat s(src[nImage]);

			CV_Assert(!s.empty());
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

	cv::Mat lutBuild(double sigma, ushort minValue, ushort maxValue)
	{
		CV_Assert(sigma >= 0);
		CV_Assert(minValue < maxValue);

		// Calculate the scale and offset
		float scale = 1.0f / float(maxValue - minValue);
		float offset = minValue;


		// Create a look-up table
		cv::Mat lut(1, 65536, CV_16UC1);
		ushort* lut_data = lut.ptr<ushort>();

		// gamma between minValue and maxValue and linear outside
		for (int i = 0; i < 65536; ++i)
		{
			if (i < minValue) {
				lut_data[i] = i;
			}
			else if (i > maxValue)
			{
				lut_data[i] = i;
			}
			else
			{
				const float normalized = (i - offset) * scale - 0.5f;
				lut_data[i] = cv::saturate_cast<ushort>(
					(1.f / (1.f + std::expf(-normalized / float(sigma)))) * float((maxValue - minValue)) + float(minValue));
			}
		}

		return lut;
	}

};
#pragma endregion

#pragma region projDenoise

class projDenoise
{
public:
	void apply(std::vector<cv::Mat>& srcDest, const cv::Mat& flatProjection, int flatBorder = 200, float sigmaSpace = 0.7f, int filterSize = 5, int level = 3)
	{
		cv::Mat flatVST;
		flatProjection.convertTo(flatVST, CV_32F);
		cv::sqrt(flatVST, flatVST);

		std::vector<float> sigmaRange = noiseEstimate(flatVST, flatBorder, level);

		MSEBilateral bFilter;

		for (auto& m : srcDest)
		{
			cv::Mat vst;
			m.convertTo(vst, CV_32F);
			cv::sqrt(vst, vst);

			bFilter.filter(vst, vst, sigmaRange, sigmaSpace, filterSize, level);

			cv::pow(vst, 2., vst);
			vst.convertTo(m, m.type());
		}
	}

private:
	std::vector<float> noiseEstimate(const cv::Mat& src, int border, int level)
	{
		std::vector<cv::Mat> destPyramid;
		cv::Mat reduced(src.rowRange(border, src.rows - border).colRange(border, src.cols - border));
		CV_Assert(reduced.total() > 0);

		buildLaplacianPyramid(reduced, destPyramid, level);

		std::vector<float> noiseSigma;
		for (const auto& m : destPyramid)
		{
			cv::Size kSize(11, 11);

			cv::Mat mean;
			cv::Mat sqMean;
			cv::pow(m, 2., sqMean);

			cv::boxFilter(m, mean, -1, kSize, cv::Point(-1, -1), true);
			cv::pow(mean, 2., mean);
			cv::boxFilter(sqMean, sqMean, -1, kSize, cv::Point(-1, -1), true);

			mean = sqMean - mean;

			cv::Mat meanWoBorder(mean.rowRange(kSize.height, mean.rows - kSize.height).colRange(kSize.width, mean.cols - kSize.width));
			cv::sqrt(meanWoBorder, meanWoBorder);

			cv::Scalar meanValue;
			cv::Scalar stdDevValue;
			cv::meanStdDev(meanWoBorder, meanValue, stdDevValue);

			noiseSigma.push_back(float(meanValue[0]));
		}

		return noiseSigma;
	}
};
#pragma endregion