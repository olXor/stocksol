#include "kernel.cuh"
#define NEGATIVE_TRANSFER_FACTOR 0.1f

__host__ __device__ float transferFunction(float in) {
	if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
		return in;
	if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
		return NEGATIVE_TRANSFER_FACTOR*in;
	return TRANSFER_WIDTH*(log(1.0f + exp(in / TRANSFER_WIDTH)) - NEGATIVE_TRANSFER_FACTOR*log(1.0f + exp(-in / TRANSFER_WIDTH)));
}

__host__ __device__ float transferDerivative(float in) {
	if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
		return 1;
	if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
		return NEGATIVE_TRANSFER_FACTOR;
	return 1.0f / (1.0f + exp(-in / TRANSFER_WIDTH)) + NEGATIVE_TRANSFER_FACTOR / (1.0f + exp(in / TRANSFER_WIDTH));
}

#ifdef MAX_WEIGHT_CHANGE
__device__ float boundChange(float change) {
	if (change > MAX_WEIGHT_CHANGE)
		change = MAX_WEIGHT_CHANGE;
	else if (change < -MAX_WEIGHT_CHANGE)
		change = -MAX_WEIGHT_CHANGE;
	return change;
}
#endif

__device__ bool isNan(float num) {
	return !isfinite(num);
}

__device__ void sumVector(float* vec, size_t size, size_t threadNum, size_t numThreads) {
	size_t stride = 1;
	while (stride < size) {
		for (size_t j = 2 * stride*threadNum; j + stride < size; j += 2 * stride*numThreads) {
			vec[j] += vec[j + stride];
		}
		stride *= 2;
		__syncthreads();
	}
}

__global__ void convolve(ConvolutionMatrices* mat, ConvolutionParameters* pars) {
	size_t outLoc = blockIdx.x;
	//size_t numOutLocBlocks = pars->forNBlockX;
	size_t inNeuron = threadIdx.x;
	size_t numInThreads = pars->forBlockX;
	size_t outNeuron = threadIdx.y;
	size_t numOutThreads = pars->forBlockY;

	size_t numOutputNeurons = pars->numOutputNeurons;
	size_t numInputNeurons = pars->numInputNeurons;
	size_t convSize = pars->convSize;
	//size_t numOutputLocs = pars->numOutputLocs;

	//SHARED
	extern __shared__ float inputs [];	//convSize*numInputNeurons
	for (size_t i = inNeuron; i < numInputNeurons; i += numInThreads) {
		for (size_t j = outNeuron; j < convSize; j += numOutThreads) {
			inputs[i + j*numInputNeurons] = mat->inlayer[i + (outLoc + j)*numInputNeurons];
		}
	}

	__syncthreads();

	//SHARED
	float* nodeStrengths = &inputs[convSize*numInputNeurons]; //numInputNeurons*numOutputNeurons
	for (size_t i = inNeuron; i < numInputNeurons; i += numInThreads) {
		for (size_t j = outNeuron; j < numOutputNeurons; j += numOutThreads) {
			nodeStrengths[i + j*numInputNeurons] = 0;
			for (size_t k = 0; k < convSize; k++) {
				nodeStrengths[i + j*numInputNeurons] += mat->weights[i + j*numInputNeurons + k*numInputNeurons*numOutputNeurons] * inputs[i + k*numInputNeurons];
			}
		}
	}

	__syncthreads();

	//note: this might fail due to improper synchronization if numOutputNeurons is not a multiple of numOutThreads
	for (size_t j = outNeuron; j < numOutputNeurons; j += numOutThreads) {
		sumVector(&nodeStrengths[j*numInputNeurons], numInputNeurons, inNeuron, numInThreads);
		if (inNeuron == 0)
			nodeStrengths[j*numInputNeurons] -= mat->outThresholds[j];
	}

	__syncthreads();

	for (size_t j = outNeuron; j < numOutputNeurons; j += numOutThreads) {
		if (inNeuron == 0)
			mat->outlayer[j + outLoc*numOutputNeurons] = mat->dropoutFactors[j + outLoc*numOutputNeurons] * transferFunction(nodeStrengths[j*numInputNeurons]);
		if (inNeuron == 1 % numInThreads)
			mat->outTDs[j + outLoc*numOutputNeurons] = mat->dropoutFactors[j + outLoc*numOutputNeurons] * transferDerivative(nodeStrengths[j*numInputNeurons]);
	}
}

__global__ void propagateErrorConvolution(ConvolutionMatrices* mat, ConvolutionParameters* pars) {
	size_t inNeuron = blockIdx.x;
	size_t inLoc = blockIdx.y;
	size_t outNeuron = threadIdx.x;
	size_t numOutNeuronThreads = pars->backPropErrBlockX;
	size_t convLoc = threadIdx.y;
	size_t numConvThreads = pars->backPropErrBlockY;

	size_t numOutputNeurons = pars->numOutputNeurons;
	size_t numInputNeurons = pars->numInputNeurons;
	size_t convSize = pars->convSize;
	size_t numOutputLocs = pars->numOutputLocs;

	extern __shared__ float inErrors[]; //convSize*numOutputNeurons

	for (size_t i = outNeuron; i < numOutputNeurons; i += numOutNeuronThreads) {
		for (size_t j = convLoc; j < convSize; j += numConvThreads) {
			if (inLoc >= j && inLoc - j < numOutputLocs)
				inErrors[i + j*numOutputNeurons] = mat->weights[inNeuron + i*numInputNeurons + j*numInputNeurons*numOutputNeurons] * mat->outErrors[i + (inLoc - j)*numOutputNeurons] * mat->outTDs[i + (inLoc - j)*numOutputNeurons];
			else
				inErrors[i + j*numOutputNeurons] = 0;
		}
	}

	__syncthreads();

	sumVector(&inErrors[0], convSize*numOutputNeurons, outNeuron + convLoc*numOutNeuronThreads, numConvThreads*numOutNeuronThreads);

	if (threadIdx.x == 0 && threadIdx.y == 0) {
		mat->inErrors[inNeuron + inLoc*numInputNeurons] = inErrors[0];
	}
}

__global__ void updateWeightsConvolution(ConvolutionMatrices* mat, ConvolutionParameters* pars) {
	size_t inNeuron = blockIdx.x;
	size_t outNeuron = blockIdx.y;
	size_t convLoc = blockIdx.z;

	size_t outLoc = threadIdx.x;
	size_t numOutLocThreads = pars->backUpdateBlockX;

	size_t numOutputLocs = pars->numOutputLocs;
	size_t numInputNeurons = pars->numInputNeurons;
	size_t numOutputNeurons = pars->numOutputNeurons;

	extern __shared__ float weightChanges[]; //numOutputLocs

	//thresholds first
	if (inNeuron == 0 && convLoc == 0) {
		for (size_t i = outLoc; i < numOutputLocs; i += numOutLocThreads) {
			weightChanges[i] = mat->outErrors[outNeuron + i*numOutputNeurons] * mat->outTDs[outNeuron + i*numOutputNeurons];
		}

		__syncthreads();

		sumVector(&weightChanges[0], numOutputLocs, outLoc, numOutLocThreads);

#ifdef BATCH_MODE
		mat->outThreshChanges[outNeuron] += weightChanges[0];
#else
#ifdef MAX_WEIGHT_CHANGE
		float change = boundChange(weightChanges[0]);
#else
		float change = weightChanges[0];
#endif
		mat->outThresholds[outNeuron] += change;
#endif
	}

	//now the weights
	for (size_t i = outLoc; i < numOutputLocs; i += numOutLocThreads) {
		weightChanges[i] = mat->inlayer[inNeuron + (i + convLoc)*numInputNeurons] * mat->outErrors[outNeuron + i*numOutputNeurons] * mat->outTDs[outNeuron + i*numOutputNeurons];
	}

	__syncthreads();

	sumVector(&weightChanges[0], numOutputLocs, outLoc, numOutLocThreads);

#ifdef BATCH_MODE
		mat->weightChanges[inNeuron + outNeuron*numInputNeurons + convLoc*numInputNeurons*numOutputNeurons] -= weightChanges[0];
#else
#ifdef MAX_WEIGHT_CHANGE
		float change = boundChange(weightChanges[0]);
#else
		float change = weightChanges[0];
#endif
		mat->weights[inNeuron + outNeuron*numInputNeurons + convLoc*numInputNeurons*numOutputNeurons] -= change;
#endif
}

__global__ void calcMaxPool(MaxPoolMatrices* mat, MaxPoolParameters* pars) {
	size_t outNeuron = blockIdx.x*blockDim.x + threadIdx.x;
	size_t outLoc = blockIdx.y*blockDim.y + threadIdx.y;

	size_t numNeurons = pars->numNeurons;
	if (outNeuron >= numNeurons || outLoc >= pars->numOutputLocs)
		return;

	size_t outIndex = outNeuron + outLoc*numNeurons;
	size_t aIndex = outNeuron + 2 * outLoc*numNeurons;
	size_t bIndex = outNeuron + (2 * outLoc + 1)*numNeurons;

	float a = mat->inlayer[aIndex];
	float b = mat->inlayer[bIndex];

	if (fabs(a) > fabs(b)) {
		mat->outlayer[outIndex] = a;
		mat->maxIndices[outIndex] = aIndex;
	}
	else {
		mat->outlayer[outIndex] = b;
		mat->maxIndices[outIndex] = bIndex;
	}
}

__global__ void bpMaxPool(MaxPoolMatrices* mat, MaxPoolParameters* pars) {
	size_t outNeuron = blockIdx.x*blockDim.x + threadIdx.x;
	size_t outLoc = blockIdx.y*blockDim.y + threadIdx.y;

	size_t numNeurons = pars->numNeurons;
	if (outNeuron >= numNeurons || outLoc >= pars->numOutputLocs)
		return;

	size_t outIndex = outNeuron + outLoc*numNeurons;
	size_t aIndex = outNeuron + 2 * outLoc*numNeurons;
	size_t bIndex = outNeuron + (2 * outLoc + 1)*numNeurons;

	if (mat->maxIndices[outIndex] == aIndex) {
		mat->inError[aIndex] = mat->outError[outIndex];
		mat->inError[bIndex] = 0;
	}
	else {
		mat->inError[aIndex] = 0;
		mat->inError[bIndex] = mat->outError[outIndex];
	}
}

__global__ void calcFixedNet(FixedNetMatrices* mat, FixedNetParameters* pars) {
	size_t inNeuron = threadIdx.x;
	size_t outNeuron = blockIdx.x;

	size_t numInThreads = pars->forBlockX;

	size_t numInputNeurons = pars->numInputNeurons;

	//SHARED
	extern __shared__ float inputs[]; //numInputNeurons;

	inputs[inNeuron] = mat->inlayer[inNeuron];

	__syncthreads();

	//SHARED
	float* outputs = &inputs[numInputNeurons]; //numInputNeurons

	outputs[inNeuron] = mat->weights[inNeuron + outNeuron*numInputNeurons] * inputs[inNeuron];

	__syncthreads();

	sumVector(outputs, numInputNeurons, inNeuron, numInThreads);

	if (inNeuron == 0) {
		float outVal = outputs[0] - mat->outThresholds[outNeuron];
		if (pars->TFOutput)
			mat->outlayer[outNeuron] = mat->dropoutFactors[outNeuron] * transferFunction(outVal);
		else
			mat->outlayer[outNeuron] = outVal;
	}

	if (inNeuron == 1 % blockDim.x) {
		if (pars->TFOutput) {
			float outVal = outputs[0] - mat->outThresholds[outNeuron];
			mat->outTDs[outNeuron] = mat->dropoutFactors[outNeuron] * transferDerivative(outVal);
		}
		else
			mat->outTDs[outNeuron] = 1;
	}
}

__global__ void bpFixedNet(FixedNetMatrices* mat, FixedNetParameters* pars) {
	size_t inNeuron = blockIdx.x;
	size_t outNeuron = threadIdx.x;

	size_t numOutThreads = pars->backBlockX;
	size_t numInputNeurons = pars->numInputNeurons;
	size_t numOutputNeurons = pars->numOutputNeurons;

	//SHARED
	extern __shared__ float outErrorTDs[]; //numOutputNeurons
	outErrorTDs[outNeuron] = mat->outErrors[outNeuron] * mat->outTDs[outNeuron];
	if (inNeuron == 0) {
#ifdef BATCH_MODE
		mat->outThreshChanges[outNeuron] += outErrorTDs[outNeuron];
#else
#ifdef MAX_WEIGHT_CHANGE
		float change = boundChange(outErrorTDs[outNeuron]);
#else
		float change = outErrorTDs[outNeuron];
#endif
		mat->outThresholds[outNeuron] += change;
#endif
	}

	__syncthreads();

	float inNeuronInput = mat->inlayer[inNeuron];

	//SHARED
	float* inerrors = &outErrorTDs[numOutputNeurons]; //numOutputNeurons

	inerrors[outNeuron] = mat->weights[inNeuron + outNeuron*numInputNeurons] * outErrorTDs[outNeuron];

#ifdef BATCH_MODE
	mat->weightChanges[inNeuron + outNeuron*numInputNeurons] -= inNeuronInput * outErrorTDs[outNeuron];
#else
#ifdef MAX_WEIGHT_CHANGE
	float change = boundChange(inNeuronInput * outErrorTDs[outNeuron]);
#else
	float change = inNeuronInput * outErrorTDs[outNeuron];
#endif
	mat->weights[inNeuron + outNeuron*numInputNeurons] -= change;
#endif

	__syncthreads();

	sumVector(inerrors, numOutputNeurons, outNeuron, numOutThreads);

	if (outNeuron == 0) {
		mat->inErrors[inNeuron] = inerrors[0];
	}
}

#ifdef BATCH_MODE
__global__ void batchUpdateConvWeights(ConvolutionMatrices* mat, ConvolutionParameters* pars) {
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;
	size_t numWeights = pars->numInputNeurons*pars->numOutputNeurons*pars->convSize;
	size_t numThresholds = pars->numOutputNeurons;

	if (i < numWeights) {
		mat->weights[i] += boundChange(mat->weightChanges[i]);
		mat->weightChanges[i] = 0;
	}

	if (i < numThresholds) {
		mat->outThresholds[i] += boundChange(mat->outThreshChanges[i]);
		mat->outThreshChanges[i] = 0;
	}
}

__global__ void batchUpdateFixedWeights(FixedNetMatrices* mat, FixedNetParameters* pars) {
	size_t i = blockIdx.x*blockDim.x + threadIdx.x;
	size_t numWeights = pars->numInputNeurons*pars->numOutputNeurons;
	size_t numThresholds = pars->numOutputNeurons;

	if (i < numWeights) {
		mat->weights[i] += boundChange(mat->weightChanges[i]);
		mat->weightChanges[i] = 0;
	}

	if (i < numThresholds) {
		mat->outThresholds[i] += boundChange(mat->outThreshChanges[i]);
		mat->outThreshChanges[i] = 0;
	}
}
#endif

__global__ void calculateOutputError(FixedNetMatrices* mat, float* stepfactor, float* correctoutput, float* hostoutput) {
	float error = *stepfactor*(mat->outlayer[threadIdx.x] - correctoutput[threadIdx.x]);
	mat->outErrors[threadIdx.x] = error;
	hostoutput[threadIdx.x] = mat->outlayer[threadIdx.x];
}

__global__ void initConvDropoutFactors(ConvolutionMatrices* mat, ConvolutionParameters* pars, size_t seed, size_t sequenceStart) {
	size_t outNeuron = threadIdx.x;
	size_t numOutNeurons = pars->numOutputNeurons;
	size_t outLoc = blockIdx.x;
	size_t seq = sequenceStart + outNeuron;
	curand_init(seed, seq, 0, &mat->randStates[outNeuron + numOutNeurons*outLoc]);
}

__global__ void initFixedDropoutFactors(FixedNetMatrices* mat, FixedNetParameters* pars, size_t seed, size_t sequenceStart) {
	size_t outNeuron = threadIdx.x;
	size_t seq = sequenceStart + outNeuron;
	curand_init(seed, seq, 0, &mat->randStates[outNeuron]);
}

__global__ void generateConvDropoutMask(ConvolutionMatrices* mat, ConvolutionParameters* pars, float dropout) {
	size_t outNeuron = threadIdx.x;
	size_t numOutNeurons = pars->numOutputNeurons;
	size_t outLoc = blockIdx.x;
	size_t dropPosition = outNeuron + numOutNeurons*outLoc;
	if (dropout > 1.0f) {
		if (curand_uniform(&mat->randStates[dropPosition]) < 1.0f / dropout)
			mat->dropoutFactors[dropPosition] = dropout;
		else
			mat->dropoutFactors[dropPosition] = 0.0f;
	}
	else {
		mat->dropoutFactors[dropPosition] = 1.0f;
	}
}

__global__ void generateFixedDropoutMask(FixedNetMatrices* mat, FixedNetParameters* pars, float dropout) {
	size_t outNeuron = threadIdx.x;
	if (dropout > 1.0f) {
		if (curand_uniform(&mat->randStates[outNeuron]) < 1.0f / dropout)
			mat->dropoutFactors[outNeuron] = dropout;
		else
			mat->dropoutFactors[outNeuron] = 0.0f;
	}
	else {
		mat->dropoutFactors[outNeuron] = 1.0f;
	}
}

size_t getConvolveSharedSize(ConvolutionParameters* pars) {
	size_t size = 0;
	size += pars->convSize*pars->numInputNeurons;	//inputs
	size += pars->numOutputNeurons*pars->numInputNeurons;	//nodeStrengths
	size *= sizeof(float);
	return size;
}

size_t getBPEConvolutionSharedSize(ConvolutionParameters* pars) {
	size_t size = 0;
	size += pars->convSize*pars->numOutputNeurons;
	size *= sizeof(float);
	return size;
}
size_t getBackUpdateConvolutionSharedSize(ConvolutionParameters* pars) {
	size_t size = 0;
	size += pars->numOutputLocs;
	size *= sizeof(float);
	return size;
}

size_t getCalcFixedSharedSize(FixedNetParameters* pars) {
	size_t size = 0;
	size += pars->numInputNeurons;	//inputs
	size += pars->numInputNeurons; //outputs
	size *= sizeof(float);
	return size;
}

size_t getBPFixedNetSharedSize(FixedNetParameters* pars) {
	size_t size = 0;
	size += pars->numOutputNeurons; //outErrorTDs
	size += pars->numOutputNeurons; //inerrors
	size *= sizeof(float);
	return size;
}
