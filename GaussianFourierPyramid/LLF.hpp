#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <intrin.h>
#include <vector>

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
	void filter(const cv::Mat& src, cv::Mat& dest, float sigma_range, float sigma_space, int filterSize, int level = 2)
	{
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
			applyBilateral(LaplacianPyramid[n], LaplacianPyramid[n], filterSize, sigma_range, sigma_space);
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
		LaplacianPyramid[level].create( GaussianPyramid[level].size(), GaussianPyramid[level].type()) ;
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
	void applyGamma(const cv::Mat& srcDest, const cv::Mat& mask, double gamma, double tailPercent = 0.001, double enlargeRangePerc = 0.1)
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
	void gammaCorrection(const std::vector<cv::Mat>& src, double gamma, ushort minValue, ushort maxValue)
	{
		cv::Mat lut = buildGammaLut(gamma, minValue, maxValue);
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

