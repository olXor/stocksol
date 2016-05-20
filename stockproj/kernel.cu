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

__device__ bool isNan(float num) {
	if (!isfinite(num))
		return false;
	return true;
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
			mat->outlayer[j + outLoc*numOutputNeurons] = transferFunction(nodeStrengths[j*numInputNeurons]);
		if (inNeuron == 1 % numInThreads)
			mat->outTDs[j + outLoc*numOutputNeurons] = transferDerivative(nodeStrengths[j*numInputNeurons]);
	}
}

__global__ void bpConvolution(ConvolutionMatrices* mat, ConvolutionParameters* pars) {
	size_t inNeuron = blockIdx.x;
	size_t numInNeuronBlocks = pars->backNBlockX;
	size_t outNeuron = threadIdx.x;
	size_t numOutNeuronThreads = pars->backBlockX;
	size_t inLoc = threadIdx.y;
	size_t numInLocThreads = pars->backBlockY;

	size_t numOutputNeurons = pars->numOutputNeurons;
	size_t numInputNeurons = pars->numInputNeurons;
	size_t numInputLocs = pars->numInputLocs;
	size_t numOutputLocs = pars->numOutputLocs;
	size_t convSize = pars->convSize;

	//SHARED
	extern __shared__ float outThreshChanges[]; //numOutputLocs*numOutputNeurons
	//SHARED
	float* outErrorTDs = &outThreshChanges[numOutputLocs*numOutputNeurons]; //numOutputLocs*numOutputNeurons
	for (size_t i = outNeuron; i < numOutputNeurons; i += numOutNeuronThreads) {
		for (size_t j = inLoc; j < numOutputLocs; j += numInLocThreads) {
			//note that the index convention on the errors differs!
			outErrorTDs[i*numOutputLocs + j] = mat->outErrors[i + j*numOutputNeurons] * mat->outTDs[i + j*numOutputNeurons];
		}
	}

	__syncthreads();

	if (inNeuron == 0) {
		for (size_t i = outNeuron; i < numOutputNeurons; i += numOutNeuronThreads) {
			for (size_t j = inLoc; j < numOutputLocs; j += numInLocThreads) {
				outThreshChanges[i*numOutputLocs + j] = outErrorTDs[i*numOutputLocs + j];
			}
		}

		__syncthreads();

		for (size_t i = outNeuron; i < numOutputNeurons; i += numOutNeuronThreads) {
			sumVector(&outThreshChanges[i*numOutputLocs], numOutputLocs, inLoc, numInLocThreads);
		}

		if (inLoc == 0) {
			for (size_t i = outNeuron; i < numOutputNeurons; i += numOutNeuronThreads) {
				mat->outThresholds[i] += outThreshChanges[i*numOutputLocs];
			}
		}
	}

	//SHARED
	float* inErrors = &outErrorTDs[numOutputLocs*numOutputNeurons]; //convSize*numOutputNeurons*numInLocThreads
	//SHARED (REUSING THE SAME MEMORY)
	float* partialWeightChanges = inErrors; //convSize*numOutputNeurons*numInLocThreads
	float* weightChanges = &partialWeightChanges[convSize*numOutputNeurons*numInLocThreads]; //convSize*numOutputNeurons
	for (size_t i = inNeuron; i < numInputNeurons; i += numInNeuronBlocks) {
		//first inErrors
		//the naming scheme for k and inLoc might be a bit confusing here.
		for (size_t inLocStart = 0; inLocStart < numInputLocs; inLocStart += numInLocThreads) {
			size_t k = inLocStart + inLoc;
			for (size_t j = outNeuron; j < numOutputNeurons; j += numOutNeuronThreads) {
				for (size_t l = 0; l < convSize; l++) {
					if (k >= l && k - l < numOutputLocs) {
						inErrors[j + l*numOutputNeurons + inLoc*numOutputNeurons*convSize] = mat->weights[i + j*numInputNeurons + l*numInputNeurons*numOutputNeurons] * outErrorTDs[j*numOutputLocs + (k - l)];
					}
					else {
						inErrors[j + l*numOutputNeurons + inLoc*numOutputNeurons*convSize] = 0;
					}
				}
			}

			__syncthreads();

			//numInputLocs must be a multiple of numInLocThreads
			sumVector(&inErrors[inLoc*numOutputNeurons*convSize], numOutputNeurons*convSize, outNeuron, numOutNeuronThreads);

			if (outNeuron == 0)
				mat->inErrors[i + k*numInputNeurons] = inErrors[inLoc*numOutputNeurons*convSize];

			__syncthreads();
		}

		//now weightChanges
		for (size_t j = outNeuron; j < numOutputNeurons; j += numOutNeuronThreads) {
			for (size_t l = inLoc; l < convSize; l += numInLocThreads) {
				weightChanges[j + l*numOutputNeurons] = 0;
			}
		}

		for (size_t locStart = 0; locStart < numInputLocs; locStart += numInLocThreads) {
			size_t k = locStart + inLoc;
			for (size_t j = outNeuron; j < numOutputNeurons; j += numOutNeuronThreads) {
				for (size_t l = 0; l < convSize; l++) {
					if (k >= l && k - l < numOutputLocs) {
						partialWeightChanges[inLoc + j*numInLocThreads + l*numInLocThreads*numOutputNeurons] = outErrorTDs[j*numOutputLocs + (k - l)] * mat->inlayer[i + k*numInputNeurons];
					}
					else {
						partialWeightChanges[inLoc + j*numInLocThreads + l*numInLocThreads*numOutputNeurons] = 0;
					}
				}
			}

			__syncthreads();

			for (size_t j = outNeuron; j < numOutputNeurons; j += numOutNeuronThreads) {
				for (size_t l = 0; l < convSize; l++) {
					sumVector(&partialWeightChanges[j*numInLocThreads + l*numInLocThreads*numOutputNeurons], numInLocThreads, inLoc, numInLocThreads);
				}
			}
			for (size_t j = outNeuron; j < numOutputNeurons; j += numOutNeuronThreads) {
				for (size_t l = inLoc; l < convSize; l += numInLocThreads) {
					weightChanges[j + l*numOutputNeurons] += partialWeightChanges[j*numInLocThreads + l*numInLocThreads*numOutputNeurons];
				}
			}
			
			__syncthreads();
		}

		for (size_t j = outNeuron; j < numOutputNeurons; j += numOutNeuronThreads) {
			for (size_t l = inLoc; l < convSize; l += numInLocThreads) {
				mat->weights[i + j*numInputNeurons + l*numInputNeurons*numOutputNeurons] -= weightChanges[j + l*numOutputNeurons];
			}
		}
	}
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
			mat->outlayer[outNeuron] = transferFunction(outVal);
		else
			mat->outlayer[outNeuron] = outVal;
	}

	if (inNeuron == 1 % blockDim.x) {
		if (pars->TFOutput) {
			float outVal = outputs[0] - mat->outThresholds[outNeuron];
			mat->outTDs[outNeuron] = transferDerivative(outVal);
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
	if (inNeuron == 0)
		mat->outThresholds[outNeuron] += outErrorTDs[outNeuron];

	__syncthreads();

	float inNeuronInput = mat->inlayer[inNeuron];

	//SHARED
	float* inerrors = &outErrorTDs[numOutputNeurons]; //numOutputNeurons

	inerrors[outNeuron] = mat->weights[inNeuron + outNeuron*numInputNeurons] * outErrorTDs[outNeuron];

	mat->weights[inNeuron + outNeuron*numInputNeurons] -= inNeuronInput * outErrorTDs[outNeuron];

	__syncthreads();

	sumVector(inerrors, numOutputNeurons, outNeuron, numOutThreads);

	if (outNeuron == 0) {
		mat->inErrors[inNeuron] = inerrors[0];
	}
}

size_t getConvolveSharedSize(ConvolutionParameters* pars) {
	size_t size = 0;
	size += pars->convSize*pars->numInputNeurons;	//inputs
	size += pars->numOutputNeurons*pars->numInputNeurons;	//nodeStrengths
	size *= sizeof(float);
	return size;
}

size_t getBPConvolutionSharedSize(ConvolutionParameters* pars) {
	size_t size = 0;
	size += pars->numOutputLocs*pars->numOutputNeurons; //outThreshChanges
	size += pars->numOutputLocs*pars->numOutputNeurons; //outErrorTDs
	size += pars->convSize*pars->numOutputNeurons*pars->backBlockY; //inErrors and partialWeightChanges
	size += pars->convSize*pars->numOutputNeurons; //weightChanges
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
