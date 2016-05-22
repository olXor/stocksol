#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include "params.h"

struct ConvolutionMatrices {
	float* inlayer;
	float* outlayer;
	float* weights;
	float* outThresholds;
#ifdef BATCH_MODE
	float* weightChanges;
	float* outThreshChanges;
#endif
	float* outTDs;
	float* inErrors;
	float* outErrors;

	size_t forwardSharedMem;
	size_t backwardSharedMem;

	size_t numInputElements;
	size_t numOutputElements;
};

struct ConvolutionParameters {
	size_t forBlockX;
	size_t forBlockY;
	size_t forNBlockX;
	size_t forNBlockY;
	size_t backBlockX;
	size_t backBlockY;
	size_t backNBlockX;
	size_t backNBlockY;

	size_t numInputLocs;
	size_t convSize;
	size_t numOutputLocs;
	size_t numInputNeurons;
	size_t numOutputNeurons;
};

struct MaxPoolMatrices {
	float* inlayer;
	float* outlayer;
	size_t* maxIndices;
	float* inError;
	float* outError;

	size_t numInputElements;
	size_t numOutputElements;
};

struct MaxPoolParameters {
	size_t blockX;
	size_t blockY;
	size_t nBlockX;
	size_t nBlockY;

	size_t numNeurons;
	size_t numInputLocs;
	size_t numOutputLocs;
};

struct FixedNetMatrices {
	float* inlayer;
	float* outlayer;
	float* weights;
	float* outThresholds;
#ifdef BATCH_MODE
	float* weightChanges;
	float* outThreshChanges;
#endif
	float* outTDs;
	float* inErrors;
	float* outErrors;

	size_t forwardSharedMem;
	size_t backwardSharedMem;
	size_t numInputElements;
	size_t numOutputElements;
};

struct FixedNetParameters {
	size_t forBlockX;
	size_t forBlockY;
	size_t forNBlockX;
	size_t forNBlockY;
	size_t backBlockX;
	size_t backBlockY;
	size_t backNBlockX;
	size_t backNBlockY;

	size_t numInputNeurons;
	size_t numOutputNeurons;

	bool TFOutput;
};

__host__ __device__ float transferFunction(float in);
__host__ __device__ float transferDerivative(float in);
__global__ void convolve(ConvolutionMatrices* mat, ConvolutionParameters* pars);
__global__ void bpConvolution(ConvolutionMatrices* mat, ConvolutionParameters* pars);

size_t getConvolveSharedSize(ConvolutionParameters* pars);
size_t getBPConvolutionSharedSize(ConvolutionParameters* pars);

__global__ void calcMaxPool(MaxPoolMatrices* mat, MaxPoolParameters* pars);
__global__ void bpMaxPool(MaxPoolMatrices* mat, MaxPoolParameters* pars);

__global__ void calcFixedNet(FixedNetMatrices* mat, FixedNetParameters* pars);
__global__ void bpFixedNet(FixedNetMatrices* mat, FixedNetParameters* pars);

#ifdef BATCH_MODE
__global__ void batchUpdateConvWeights(ConvolutionMatrices* mat, ConvolutionParameters* pars);
__global__ void batchUpdateFixedWeights(FixedNetMatrices* mat, FixedNetParameters* pars);
#endif

size_t getCalcFixedSharedSize(FixedNetParameters* pars);
size_t getBPFixedNetSharedSize(FixedNetParameters* pars);