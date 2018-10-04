#include "cpuwindow.h"
#include "cpucommon.h"
#include "cpucompute.h"
#include "cpufeature.h"

struct ExplicitDataInfo {
	std::string filelist;
	size_t numPreColumns;
	size_t waveformSize;
	std::string weightfolder;
	std::ofstream* outfile;
	int inputOffset;
	size_t headerSize = 0;
};

void saveFileResults(std::string fname, ExplicitDataInfo dataInfo, size_t* segmentNum);
void saveExplicitBinaryFileResults(std::string binfname, ExplicitDataInfo dataInfo, size_t* segmentNum, bool generateSecondaryFeatures, std::vector<float>* featureMeans, std::vector<float>* featureStdevs, bool print);

void cpuwindow() {
	ExplicitDataInfo dataInfo;

	std::cout << "Enter name of file list to test on: ";
	std::cin >> dataInfo.filelist;

	bool useConvFeatureGeneration = false;
	std::cout << "Generate convolution features? ";
	std::cin >> useConvFeatureGeneration;

	bool printOutputs = false;
	bool binaryInput = false;
	if (useConvFeatureGeneration) {
		dataInfo.numPreColumns = 16;
		dataInfo.waveformSize = 1024;	//double the actual waveform size, since we assume the input file includes peak data as well (1 for peak, -1 for valley, 0 otherwise; located after the waveform data)
		dataInfo.inputOffset = 0;
		binaryInput = true;
		dataInfo.headerSize = 8;
	}
	else {
		std::cout << "Enter number of information columns (coming before the waveform data): ";
		std::cin >> dataInfo.numPreColumns;

		std::cout << "Enter waveform size: ";
		std::cin >> dataInfo.waveformSize;

		std::cout << "Enter input offset (-1 for random offset): ";
		std::cin >> dataInfo.inputOffset;

		std::cout << "Are the input files binary? ";
		std::cin >> binaryInput;

		if (binaryInput)  {
			std::cout << "Enter header size: ";
			std::cin >> dataInfo.headerSize;
		}
	}

	std::cout << "Enter weight folder: ";
	std::cin >> dataInfo.weightfolder;

	std::cout << "Print outputs to console? ";
	std::cin >> printOutputs;

	std::string outfname;
	std::cout << "Enter output file name: ";
	std::cin >> outfname;
	std::ofstream outfile(savestring + dataInfo.weightfolder + "/" + outfname);
	dataInfo.outfile = &outfile;

	loadArchitecture(savestring + dataInfo.weightfolder + "/archWeights");
	if (!loadWeights(weightlayers, savestring + dataInfo.weightfolder + "/weights_bin"))
		throwError("Couldn't open weights file");
	if (!loadBatchNormData(weightlayers, savestring + dataInfo.weightfolder + "/weights_batchnorm_bin")) {
		throwError("Couldn't open batchnorm file");
	}
	std::vector<float> featureMeans;
	std::vector<float> featureStdevs;
	if (useConvFeatureGeneration) {
		if (!loadFeatureNorms(savestring + dataInfo.weightfolder + "/weights_featurenorms", &featureMeans, &featureStdevs))
			throwError("Couldn't open featurenorm file");
	}
	loadOutputScalings(savestring + dataInfo.weightfolder + "/weights_outscale");

	std::ifstream filelist(datastring + dataInfo.filelist);
	std::string line;
	size_t segmentNum = 0;
	while (std::getline(filelist, line)) {
		std::stringstream lss(line);
		std::string fname;
		lss >> fname;
		std::cout << "Testing file " << fname << ": ";
		if (!binaryInput)
			saveFileResults(datastring + fname, dataInfo, &segmentNum);
		else
			saveExplicitBinaryFileResults(datastring + fname, dataInfo, &segmentNum, useConvFeatureGeneration, &featureMeans, &featureStdevs, printOutputs);
		std::cout << " Done. " << std::endl;
	}
}

void saveFileResults(std::string fname, ExplicitDataInfo dataInfo, size_t* segmentNum) {
	std::ifstream infile(fname);
	std::string line;
	std::vector<std::string> precolumns(dataInfo.numPreColumns);
	std::vector<float> waveform(dataInfo.waveformSize);
	std::getline(infile, line);	//header
	while (std::getline(infile, line)) {
		std::stringstream lss(line);
		for (size_t c = 0; c < dataInfo.numPreColumns; c++) {
			std::string pre;
			lss >> pre;
			precolumns[c] = pre;
		}
		for (size_t c = 0; c < dataInfo.waveformSize; c++) {
			float val;
			lss >> val;
			waveform[c] = val;
		}

		if (NUM_INPUTS > dataInfo.waveformSize || (dataInfo.inputOffset > 0 && NUM_INPUTS + dataInfo.inputOffset > dataInfo.waveformSize)) {
			throwError("numInputs + inputOffset > waveformSize");
		}

		size_t offset = 0;
		if (dataInfo.inputOffset >= 0)
			offset = dataInfo.inputOffset;
		else
			offset = rand() % (dataInfo.waveformSize - NUM_INPUTS + 1);

		float minVal = 999999;
		float maxVal = -999999;
		for (size_t c = offset; c < NUM_INPUTS + offset; c++) {
			minVal = std::min(minVal, waveform[c]);
			maxVal = std::max(maxVal, waveform[c]);
		}
		for (size_t c = offset; c < NUM_INPUTS + offset; c++) {
			if (maxVal > minVal)
				waveform[c] = 2.0f*(waveform[c] - minVal) / (maxVal - minVal) - 1.0f;
			else
				waveform[c] = 0.0f;
		}

		//---calculate---
		memcpy(&weightlayers[0]->inlayer[0], &waveform[offset], NUM_INPUTS*sizeof(float));

		for (size_t l = 0; l < weightlayers.size(); l++) {
			weightlayers[l]->calc();
		}
		//---end calculate---

		float* outlayer = weightlayers[weightlayers.size() - 1]->outlayer;
		float output = outlayer[0];
		if (outputMeanScale.size() > 0 && outputStdevScale.size() > 0)
			output = output*outputStdevScale[0] + outputMeanScale[0];
		(*dataInfo.outfile) << (*segmentNum) << " " << output << " " << fname << " ";
		for (size_t c = 0; c < dataInfo.numPreColumns; c++) {
			(*dataInfo.outfile) << precolumns[c] << " ";
		}
		(*dataInfo.outfile) << std::endl;

		(*segmentNum)++;

	}
}

void saveExplicitBinaryFileResults(std::string binfname, ExplicitDataInfo dataInfo, size_t* segmentNum, bool generateSecondaryFeatures, std::vector<float>* featureMeans, std::vector<float>* featureStdevs, bool print) {
	if (!generateSecondaryFeatures && dataInfo.waveformSize != weightlayers[0]->numInputElements)
		throwError("Invalid input size");

	FILE* binfile = fopen(binfname.c_str(), "rb");
	std::string line;
	std::vector<float> columns(dataInfo.numPreColumns + dataInfo.waveformSize);
	std::vector<float> secondaryFeatures;
	_fseeki64(binfile, dataInfo.headerSize, SEEK_SET);
	
	while (fread(&columns[0], sizeof(float), dataInfo.numPreColumns + dataInfo.waveformSize, binfile) == dataInfo.numPreColumns + dataInfo.waveformSize) {
		//---calculate---
		if (generateSecondaryFeatures) {
			createSecondaryFeatures(&columns[dataInfo.numPreColumns], &columns[dataInfo.numPreColumns + dataInfo.waveformSize / 2], &secondaryFeatures, featureMeans, featureStdevs);
			if (secondaryFeatures.size() != weightlayers[0]->numInputElements)
				throwError("Generated feature size doesn't match network input size");
			memcpy(&weightlayers[0]->inlayer[0], &secondaryFeatures[0], secondaryFeatures.size()*sizeof(float));
		}
		else {
			memcpy(&weightlayers[0]->inlayer[0], &columns[dataInfo.numPreColumns], dataInfo.waveformSize*sizeof(float));
		}

		for (size_t l = 0; l < weightlayers.size(); l++) {
			weightlayers[l]->calc();
		}
		//---end calculate---

		float* outlayer = weightlayers[weightlayers.size() - 1]->outlayer;
		size_t numOutputs = weightlayers[weightlayers.size() - 1]->numOutputElements;
		(*dataInfo.outfile) << (*segmentNum) << " " <<  binfname << " ";
		if (print)
			std::cout << (*segmentNum) << " " << binfname << " " << std::endl;
		for (size_t i = 0; i < numOutputs; i++) {
			float output = outlayer[i];
			if (outputMeanScale.size() > i && outputStdevScale.size() > i)
				output = output*outputStdevScale[i] + outputMeanScale[i];
			(*dataInfo.outfile) << output << " ";
			if (print)
				std::cout << output << " ";
		}
		if (print)
			std::cout << std::endl;
		for (size_t p = 0; p < dataInfo.numPreColumns; p++) {
			(*dataInfo.outfile) << columns[p] << " ";
			if (print)
				std::cout << columns[p] << " ";
		}
		(*dataInfo.outfile) << std::endl;
		if (print) {
			std::cout << std::endl;
			system("pause");
		}

		(*segmentNum)++;
	}
}
