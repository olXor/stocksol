#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include "params.h"
#include <curand_kernel.h>

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

	curandState* randStates;
	float* dropoutFactors;

	size_t forwardSharedMem;
	size_t backPropErrSharedMem;
	size_t backUpdateSharedMem;

	size_t numInputElements;
	size_t numOutputElements;
};

struct ConvolutionParameters {
	size_t forBlockX;
	size_t forBlockY;
	size_t forNBlockX;
	size_t forNBlockY;
	size_t backPropErrBlockX;	//outNeuron
	size_t backPropErrBlockY;	//convLoc
	size_t backPropErrNBlockX;	//inNeuron
	size_t backPropErrNBlockY;	//inLoc
	size_t backUpdateBlockX;	//outLoc
	size_t backUpdateNBlockX;	//inNeuron;
	size_t backUpdateNBlockY;	//outNeuron;
	size_t backUpdateNBlockZ;	//convLoc

	size_t numInputLocs;
	size_t convSize;
	size_t numOutputLocs;
	size_t numInputNeurons;
	size_t numOutputNeurons;

	size_t transferType = TRANSFER_TYPE_RECTIFIER;
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

	curandState* randStates;
	float* dropoutFactors;

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

	size_t transferType = TRANSFER_TYPE_RECTIFIER;

	bool TFOutput = true;
};

__global__ void convolve(ConvolutionMatrices* mat, ConvolutionParameters* pars);
__global__ void propagateErrorConvolution(ConvolutionMatrices* mat, ConvolutionParameters* pars);
__global__ void updateWeightsConvolution(ConvolutionMatrices* mat, ConvolutionParameters* pars);

size_t getConvolveSharedSize(ConvolutionParameters* pars);
size_t getBPEConvolutionSharedSize(ConvolutionParameters* pars);
size_t getBackUpdateConvolutionSharedSize(ConvolutionParameters* pars);

__global__ void calcMaxPool(MaxPoolMatrices* mat, MaxPoolParameters* pars);
__global__ void bpMaxPool(MaxPoolMatrices* mat, MaxPoolParameters* pars);

__global__ void calcFixedNet(FixedNetMatrices* mat, FixedNetParameters* pars);
__global__ void bpFixedNet(FixedNetMatrices* mat, FixedNetParameters* pars);

__global__ void calculateOutputError(FixedNetMatrices* mat, float* stepfactor, float* correctoutput, float* hostoutput);

__global__ void initConvDropoutFactors(ConvolutionMatrices* mat, ConvolutionParameters* pars, size_t seed, size_t sequenceStart);
__global__ void initFixedDropoutFactors(FixedNetMatrices* mat, FixedNetParameters* pars, size_t seed, size_t sequenceStart);
__global__ void generateConvDropoutMask(ConvolutionMatrices* mat, ConvolutionParameters* pars, float dropout);
__global__ void generateFixedDropoutMask(FixedNetMatrices* mat, FixedNetParameters* pars, float dropout);

#ifdef BATCH_MODE
__global__ void batchUpdateConvWeights(ConvolutionMatrices* mat, ConvolutionParameters* pars);
__global__ void batchUpdateFixedWeights(FixedNetMatrices* mat, FixedNetParameters* pars);
#endif

size_t getCalcFixedSharedSize(FixedNetParameters* pars);
size_t getBPFixedNetSharedSize(FixedNetParameters* pars);