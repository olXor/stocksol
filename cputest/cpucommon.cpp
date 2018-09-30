#include "cpucommon.h"

float mean(std::vector<float> in) {
	float mean = 0;
	for (size_t i = 0; i < in.size(); i++) {
		mean += in[i];
	}
	mean /= in.size();
	return mean;
}

float stdev(std::vector<float> in, float mean) {
	float stdev = 0;
	for (size_t i = 0; i < in.size(); i++) {
		stdev += pow(in[i] - mean, 2);
	}
	stdev /= in.size();
	stdev = sqrt(stdev);
	return stdev;
}

void throwError(std::string err) {
	std::cout << err << std::endl;
	throw std::runtime_error(err);
}

void loadArchitecture(std::string archfname) {
	std::ifstream archfile(archfname);

	if (!archfile.is_open())
		throwError("Couldn't open architecture file");

	std::string line;

	while (std::getline(archfile, line)) {
		std::stringstream lss(line);

		std::string layerType;
		lss >> layerType;

		Layer* newLayer = NULL;

		if (layerType == "Entry") {
			std::string name;
			size_t nInputs;
			if (!(lss >> name >> nInputs))
				throwError("Invalid parameters for Entry layer " + name);

			newLayer = NULL;
			NUM_INPUTS = nInputs;
		}
		else if (layerType == "Fixed") {
			std::string name;
			size_t connection;
			size_t nInputs;
			size_t nOutputs;
			if (!(lss >> name >> connection >> nOutputs))
				throwError("Invalid parameters for Fixed layer " + name);

			if (connection == 1)
				nInputs = NUM_INPUTS;
			else if (connection - 2 < weightlayers.size())
				nInputs = weightlayers[connection - 2]->numOutputElements;
			else
				throwError("Invalid connection for layer " + name);

			newLayer = new FixedLayer(nInputs, nOutputs, TRANSFER_TYPE_RECTIFIER);
			weightlayers.push_back(newLayer);
			if (connection > 1)
				newLayer->link(weightlayers[connection - 2]);
		}
		else if (layerType == "Convolution") {
			std::string name;
			size_t con;
			size_t inLocs;
			size_t inNeurons;
			size_t outLocs;
			size_t outNeurons;
			size_t convSize;
			size_t stride;
			if (!(lss >> name >> con >> outLocs >> outNeurons >> convSize >> stride))
				throwError("Invalid parameters for convolution layer " + name);

			if (con == 1) {
				inNeurons = 1;
				inLocs = NUM_INPUTS;
			}
			else if (con - 2 < weightlayers.size()) {
				std::vector<size_t> linkSymmetryDimensions = weightlayers[con - 2]->getOutputSymmetryDimensions();
				if (linkSymmetryDimensions.size() == 0 || linkSymmetryDimensions.size() > 2)
					throwError("Invalid linkSymmetryDimensions for ConvolutionLayer " + name + " input");
				inLocs = linkSymmetryDimensions[0];
				if (linkSymmetryDimensions.size() == 2)
					inNeurons = linkSymmetryDimensions[1];
				else
					inNeurons = 1;
			}
			else
				throwError("Invalid connection for layer " + name);

			newLayer = new ConvolutionLayer(inNeurons, inLocs, outNeurons, outLocs, convSize, stride, TRANSFER_TYPE_RECTIFIER);
			weightlayers.push_back(newLayer);
			if (con > 1)
				newLayer->link(weightlayers[con - 2]);
		}
		else if (layerType == "MaxPool") {
			std::string name;
			size_t con;
			size_t outLocs;
			size_t outNeurons;
			if (!(lss >> name >> con))
				throwError("Invalid parameters for MaxPool layer " + name);

			if (con - 2 < weightlayers.size()) {
				std::vector<size_t> linkSymmetryDimensions = weightlayers[con - 2]->getOutputSymmetryDimensions();
				if (linkSymmetryDimensions.size() == 0 || linkSymmetryDimensions.size() > 2)
					throwError("Invalid linkSymmetryDimensions for MaxPoolLayer " + name + " input");
				outLocs = linkSymmetryDimensions[0] / 2;
				if (linkSymmetryDimensions.size() == 2)
					outNeurons = linkSymmetryDimensions[1];
				else
					outNeurons = 1;
			}
			else
				throwError("Invalid connection for layer " + name);

			newLayer = new MaxPoolLayer(outNeurons, 2 * outLocs);
			weightlayers.push_back(newLayer);
			if (con > 1)
				newLayer->link(weightlayers[con - 2]);
		}
		else if (layerType == "BatchNorm") {
			std::string name;
			size_t con;
			if (!(lss >> name >> con))
				throwError("Invalid parameters for BatchNorm layer " + name);

			size_t numNeurons;
			size_t numLocsX;
			size_t numLocsY;
			if (con - 2 < weightlayers.size()) {
				std::vector<size_t> linkSymmetryDimensions = weightlayers[con - 2]->getOutputSymmetryDimensions();
				if (linkSymmetryDimensions.size() == 0 || linkSymmetryDimensions.size() > 3)
					throwError("Invalid linkSymmetryDimensions for BatchNormLayer " + name);
				else if (linkSymmetryDimensions.size() == 1) {
					numNeurons = linkSymmetryDimensions[0];
					numLocsX = 1;
					numLocsY = 1;
				}
				else if (linkSymmetryDimensions.size() == 2) {
					numNeurons = linkSymmetryDimensions[1];
					numLocsX = linkSymmetryDimensions[0];
					numLocsY = 1;
				}
				else if (linkSymmetryDimensions.size() == 3) {
					numNeurons = linkSymmetryDimensions[2];
					numLocsX = linkSymmetryDimensions[0];
					numLocsY = linkSymmetryDimensions[1];
				}
			}
			else
				throwError("Invalid connection for layer " + name);

			newLayer = new BatchNormLayer(numNeurons, numLocsX, numLocsY, TRANSFER_TYPE_RECTIFIER);
			weightlayers.push_back(newLayer);
			if (con > 1)
				newLayer->link(weightlayers[con - 2]);
		}
		else
			throwError("Unrecognized layer type: " + layerType);

		std::string flag;
		while (lss >> flag) {
			if (newLayer == NULL)
				continue;
			if (flag == "TransferType") {
				std::string type;
				lss >> type;
				if (type == "Identity")
					newLayer->changeTransferType(TRANSFER_TYPE_IDENTITY);
				else if (type == "Rectifier")
					newLayer->changeTransferType(TRANSFER_TYPE_RECTIFIER);
				else if (type == "Sigmoid")
					newLayer->changeTransferType(TRANSFER_TYPE_SIGMOID);
				else if (type == "TANH")
					newLayer->changeTransferType(TRANSFER_TYPE_TANH);
			}
			else if (flag == "CustomDeviceChainEndpoint") {
			}
			else if (flag == "InternalChainEndpoint") {
				size_t correctLayer;
				size_t correctOffset;
				lss >> correctLayer >> correctOffset;
			}
			else if (flag == "Dropout") {
				if (layerType != "Fixed")
					throwError("Dropout only allowed for the following layer types: Fixed");
				float dropFactor = 1.0f;
				lss >> dropFactor;
			}
		}
	}
}

bool loadWeights(std::vector<Layer*> layers, std::string fname) {
	std::stringstream fss;
	fss << fname;

	FILE* infile = fopen(fss.str().c_str(), "rb");

	if (infile == NULL) {
		std::cout << "Couldn't open weights file" << std::endl;
		throw new std::runtime_error("");
	}

	for (size_t lay = 0; lay < layers.size(); lay++) {
		layers[lay]->loadWeights(infile);
	}

	return true;
}

bool loadBatchNormData(std::vector<Layer*> layers, std::string fname) {
	std::stringstream fss;
	fss << fname;

	FILE* infile = fopen(fss.str().c_str(), "rb");

	if (infile == NULL) {
		std::cout << "Couldn't open batchnorm file" << std::endl;
		throw new std::runtime_error("");
	}

	for (size_t lay = 0; lay < layers.size(); lay++) {
		layers[lay]->loadBatchNormData(infile);
	}

	return true;
}

bool loadOutputScalings(std::string fname) {
	std::ifstream scalefile(fname);

	if (!scalefile.is_open())
		return false;

	std::string line;
	std::getline(scalefile, line);
	float val;
	std::stringstream stdevss(line);
	outputStdevScale.clear();
	while (stdevss >> val)
		outputStdevScale.push_back(val);

	std::getline(scalefile, line);
	outputMeanScale.clear();
	std::stringstream meanss(line);
	while (meanss >> val)
		outputMeanScale.push_back(val);
	return true;
}