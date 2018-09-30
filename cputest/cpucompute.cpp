#include "cpucompute.h"

float transferFunction(float in, size_t type) {
	if (type == TRANSFER_TYPE_RECTIFIER) {
		if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
			return in;
		if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
			return NEGATIVE_TRANSFER_FACTOR*in;
		return TRANSFER_WIDTH*(log(1.0f + exp(in / TRANSFER_WIDTH)) - NEGATIVE_TRANSFER_FACTOR*log(1.0f + exp(-in / TRANSFER_WIDTH)));
	}
	else if (type == TRANSFER_TYPE_SIGMOID) {
		if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
			return 1.0f;
		if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
			return 0.0f;
		return 1.0f / (1.0f + exp(-in / TRANSFER_WIDTH));
	}
	else if (type == TRANSFER_TYPE_TANH) {
		if (in / TRANSFER_WIDTH > TRANSFER_FUNCTION_LIMIT)
			return 1.0f;
		if (in / TRANSFER_WIDTH < -TRANSFER_FUNCTION_LIMIT)
			return -1.0f;
		float expo = exp(in / TRANSFER_WIDTH);
		return (expo - 1) / (expo + 1);
	}
	else if (type == TRANSFER_TYPE_IDENTITY)
		return in;
	return 99999999.0f;
}

void FixedLayer::calc() {
	for (size_t i = 0; i < numOutputNeurons; i++) {
		outlayer[i] = 0;
		for (size_t j = 0; j < numInputNeurons; j++) {
			outlayer[i] += weights[j + i*numInputNeurons] * inlayer[j];
		}
		outlayer[i] -= outThresholds[i];
		outlayer[i] = transferFunction(outlayer[i], transferType);
	}
}

void ConvolutionLayer::calc() {
	for (size_t oN = 0; oN < numOutputNeurons; oN++) {
		for (size_t oL = 0; oL < numOutputLocs; oL++) {
			float res = 0;
			for (size_t iN = 0; iN < numInputNeurons; iN++) {
				for (size_t c = 0; c < convSize; c++) {
					res += weights[iN + oN*numInputNeurons + c*numInputNeurons*numOutputNeurons] * inlayer[iN + (stride*oL + c)*numInputNeurons];
				}
			}
			res -= outThresholds[oN];
			outlayer[oN + oL*numOutputNeurons] = transferFunction(res, transferType);
		}
	}
}

void BatchNormLayer::calc() {
	for (size_t n = 0; n < numNeurons; n++) {
		for (size_t x = 0; x < numLocsX; x++) {
			for (size_t y = 0; y < numLocsY; y++) {
				size_t pos = n + x*numNeurons + y*numNeurons*numLocsX;
				outlayer[pos] = ((inlayer[pos] - batchMeans[n]) / batchStdevs[n])*stdevAdjusts[n] - thresholds[n];
				outlayer[pos] = transferFunction(outlayer[pos], transferType);
			}
		}
	}
}

ConvolutionLayer::ConvolutionLayer(size_t nInputNeurons, size_t nInputLocs, size_t nOutputNeurons, size_t nOutputLocs, size_t cSize, size_t newStride, size_t transType) {
	if (nInputLocs != newStride*nOutputLocs + cSize - newStride) {
		std::cout << "Invalid ConvolutionLayer parameter: nInputLocs: " << nInputLocs << " nOutputLocs: " << nOutputLocs << " convSize: " << cSize << " stride: " << newStride << std::endl;
		system("pause");
		throw new std::runtime_error("Invalid convolution parameters");
	}

	numInputNeurons = nInputNeurons;
	numInputLocs = nInputLocs;
	numOutputNeurons = nOutputNeurons;
	numOutputLocs = nOutputLocs;
	convSize = cSize;
	stride = newStride;

	inlayer = new float[numInputNeurons*numInputLocs];
	outlayer = new float[numOutputNeurons*numOutputLocs];
	weights = new float[numInputNeurons*numOutputNeurons*convSize];
	outThresholds = new float[numOutputNeurons];

	numInputElements = numInputNeurons*numInputLocs;
	numOutputElements = numOutputNeurons*numOutputLocs;

	transferType = transType;
}

ConvolutionLayer::~ConvolutionLayer() {
	delete[] inlayer;
	delete[] outlayer;
	delete[] weights;
	delete[] outThresholds;
}

FixedLayer::FixedLayer(size_t nInputNeurons, size_t nOutputNeurons, size_t transType) {
	numInputNeurons = nInputNeurons;
	numOutputNeurons = nOutputNeurons;

	inlayer = new float[nInputNeurons];
	outlayer = new float[nOutputNeurons];
	weights = new float[nInputNeurons*nOutputNeurons];
	outThresholds = new float[nOutputNeurons];

	numInputElements = numInputNeurons;
	numOutputElements = numOutputNeurons;

	transferType = transType;
}

FixedLayer::~FixedLayer() {
	delete[] inlayer;
	delete[] outlayer;
	delete[] weights;
	delete[] outThresholds;
}

void Layer::link(Layer* l) {
	if (l->numOutputElements != numInputElements) {
		std::cout << "Invalid layer link: l->numOutputElements: " << l->numOutputElements << " numInputElements: " << numInputElements << std::endl;
		throw std::runtime_error("");
	}

	delete[] inlayer;
	inlayer = l->outlayer;
}

MaxPoolLayer::MaxPoolLayer(size_t nInputNeurons, size_t nInputLocs) {
	if (nInputLocs != (nInputLocs / 2) * 2) {
		std::cout << "MaxPoolLayer with odd numInputLocs" << std::endl;
		throw std::runtime_error("");
	}

	numInputNeurons = nInputNeurons;
	numInputLocs = nInputLocs;
	numInputElements = nInputNeurons*nInputLocs;
	numOutputElements = numInputElements / 2;
	inlayer = new float[numInputElements];
	outlayer = new float[numOutputElements];
}

MaxPoolLayer::~MaxPoolLayer() {
	delete[] inlayer;
	delete[] outlayer;
}

BatchNormLayer::BatchNormLayer(size_t nNeurons, size_t nLocsX, size_t nLocsY, size_t transType) {
	numNeurons = nNeurons;
	numLocsX = nLocsX;
	numLocsY = nLocsY;

	stdevAdjusts = new float[numNeurons];
	thresholds = new float[numNeurons];
	batchStdevs = new float[numNeurons];
	batchMeans = new float[numNeurons];

	numInputElements = numNeurons*numLocsX*numLocsY;
	numOutputElements = numNeurons*numLocsX*numLocsY;
	inlayer = new float[numInputElements];
	outlayer = new float[numOutputElements];

	transferType = transType;
}

BatchNormLayer::~BatchNormLayer() {
	delete[] stdevAdjusts;
	delete[] thresholds;
	delete[] batchStdevs;
	delete[] batchMeans;
	delete[] inlayer;
	delete[] outlayer;
}

void MaxPoolLayer::calc() {
	for (size_t n = 0; n < numInputNeurons; n++)
		for (size_t i = 0; i < numInputLocs / 2; i++) {
			float a = inlayer[n + 2 * i*numInputNeurons];
			float b = inlayer[n + (2 * i + 1)*numInputNeurons];
		if (fabs(a) > fabs(b))
			outlayer[n + i*numInputNeurons] = a;
		else
			outlayer[n + i*numInputNeurons] = b;
	}
}

void ConvolutionLayer::loadWeights(FILE* infile) {
	for (size_t i = 0; i < numOutputNeurons; i++) {
		fread(&outThresholds[i], sizeof(float), 1, infile);
		for (size_t j = 0; j < numInputNeurons; j++) {
			for (size_t k = 0; k < convSize; k++) {
				fread(&weights[j + i*numInputNeurons + k*numInputNeurons*numOutputNeurons], sizeof(float), 1, infile);
			}
		}
	}
}

void MaxPoolLayer::loadWeights(FILE* infile) {
	return;
}

void FixedLayer::loadWeights(FILE* infile) {
	for (size_t i = 0; i < numOutputNeurons; i++) {
		fread(&outThresholds[i], sizeof(float), 1, infile);
		for (size_t j = 0; j < numInputNeurons; j++) {
			fread(&weights[j + i*numInputNeurons], sizeof(float), 1, infile);
		}
	}
}

void BatchNormLayer::loadWeights(FILE* infile) {
	for (size_t i = 0; i < numNeurons; i++) {
		fread(&stdevAdjusts[i], sizeof(float), 1, infile);
	}
	for (size_t i = 0; i < numNeurons; i++) {
		fread(&thresholds[i], sizeof(float), 1, infile);
	}
}

void Layer::loadBatchNormData(FILE* infile) {}

void BatchNormLayer::loadBatchNormData(FILE* infile) {
	for (size_t i = 0; i < numNeurons; i++) {
		fread(&batchStdevs[i], sizeof(float), 1, infile);
	}
	for (size_t i = 0; i < numNeurons; i++) {
		fread(&batchMeans[i], sizeof(float), 1, infile);
	}
}

std::vector<size_t> Layer::getOutputSymmetryDimensions() {
	std::vector<size_t> dim(1);
	dim[0] = numOutputElements;
	return dim;
}

std::vector<size_t> ConvolutionLayer::getOutputSymmetryDimensions() {
	std::vector<size_t> dim;
	if (numOutputLocs > 1)
		dim.push_back(numOutputLocs);
	dim.push_back(numOutputNeurons);

	return dim;
}

std::vector<size_t> MaxPoolLayer::getOutputSymmetryDimensions() {
	std::vector<size_t> dim;
	if (numInputLocs/2 > 1)
		dim.push_back(numInputLocs/2);
	dim.push_back(numInputNeurons);

	return dim;
}

std::vector<size_t> BatchNormLayer::getOutputSymmetryDimensions() {
	std::vector<size_t> dim;
	if (numLocsX > 1 || numLocsY > 1)
		dim.push_back(numLocsX);
	if (numLocsY > 1)
		dim.push_back(numLocsY);
	dim.push_back(numNeurons);
	return dim;
}

void Layer::changeTransferType(size_t newType) {
	transferType = newType;
}