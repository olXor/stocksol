#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "cpucompute.h"
#include <fstream>
#include <list>

#define savestring "saveweights/"
#define datastring "rawdata/"

#define NUM_INPUTS 136

//super quick and dirty cpu tester

std::vector<std::vector<float>> dataset;
size_t NUM_NEURONS = 32;
size_t numFixedHiddenNeurons = 1024;

std::string savename;

void loadParameters(std::string parName);
void createWeightLayers();
bool loadWeights(std::vector<Layer*> layers, std::string fname);

std::string datafname = "trainset";
std::string resultfname = "calcresults";
std::string weightfname = "weights";
size_t intervalSize = 1000;
std::vector<size_t> columns;
size_t begin = 1;
size_t end = 0;
bool discard = 1;
bool keepExtraData = 1;
std::vector<std::string> extraData;
bool usingIntervalFile = false;
bool discardIntervalRemainder = true;

size_t currentLineNum = 1;

std::vector<Layer*> weightlayers;

std::vector<std::list<float>> inputs;

bool readIntervalParameters(std::ifstream* intervalfile);
bool readIntervalData(std::ifstream* datafile, std::vector<std::vector<std::vector<float>>>* intervalData, size_t* iBegin, size_t* iEnd);
void saveIntervalResult(std::ofstream* resultfile, std::vector<std::vector<std::vector<float>>>* intervalData, bool print);
bool discardInput(float* inputs);
float mean(std::vector<float> in);
float stdev(std::vector<float> in, float mean);

#ifdef DEBUG_SAVE_INTERNAL_RESULTS
void debugSaveInternalResults();
#endif

int main() {
	loadParameters("pars.cfg");

	columns.push_back(1);
	columns.push_back(2);
	columns.push_back(3);
	columns.push_back(4);

	bool print = true;
	std::string intervalfname = "";
	std::string line;

	std::cout << "Specify interval file (blank: enter a single interval manually): ";
	std::getline(std::cin, line);
	if (!line.empty()) std::stringstream(line) >> intervalfname;
	std::cout << std::endl;

	usingIntervalFile = (intervalfname != "");

	if (!usingIntervalFile) {
		std::cout << "Enter the name of the data file to read (default: \"trainset\"): ";
		std::getline(std::cin, line);
		if (!line.empty()) std::stringstream(line) >> datafname;
		std::cout << std::endl;

		std::string colstr;
		std::cout << "Enter the column numbers (default: 1 2 3 4): ";
		std::getline(std::cin, line);
		colstr = line;
		std::cout << std::endl;

		if (colstr != "") {
			columns.clear();
			std::stringstream colss(colstr);
			size_t col;
			while (colss >> col) {
				columns.push_back(col);
			}
		}

		std::cout << "Enter the first line number to read (default: 1): ";
		std::getline(std::cin, line);
		if (!line.empty()) std::stringstream(line) >> begin;
		std::cout << std::endl;

		std::cout << "Enter the last line number to read (default: 0): ";
		std::getline(std::cin, line);
		if (!line.empty()) std::stringstream(line) >> end;
		std::cout << std::endl;
	}

	std::cout << "Enter the name of the result file to write to (default: \"calcresults\"): ";
	std::getline(std::cin, line);
	if (!line.empty()) std::stringstream(line) >> resultfname;
	std::cout << std::endl;

	std::cout << "Enter the name of the weights file (default: \"weights\"): ";
	std::getline(std::cin, line);
	if (!line.empty()) std::stringstream(line) >> weightfname;
	std::cout << std::endl;

	std::cout << "Enter the interval size (default: 1000): ";
	std::getline(std::cin, line);
	if (!line.empty()) std::stringstream(line) >> intervalSize;
	std::cout << std::endl;

	std::cout << "Output results to screen? (default: 1): ";
	std::getline(std::cin, line);
	if (!line.empty()) std::stringstream(line) >> print;
	std::cout << std::endl;

	std::cout << "Discard samples? (default: 1): ";
	std::getline(std::cin, line);
	if (!line.empty()) std::stringstream(line) >> discard;
	std::cout << std::endl;

	std::cout << "Discard interval remainder? (default: 1): ";
	std::getline(std::cin, line);
	if (!line.empty()) std::stringstream(line) >> discardIntervalRemainder;
	std::cout << std::endl;

	if (usingIntervalFile) {
		std::cout << "Keep extra data columns? (default: 1): ";
		std::getline(std::cin, line);
		if (!line.empty()) std::stringstream(line) >> keepExtraData;
		std::cout << std::endl;
	}
	else
		keepExtraData = false;

	createWeightLayers();
	if (!loadWeights(weightlayers, weightfname)) {
		std::cout << "Couldn't open weights file" << std::endl;
		system("pause");
		return 0;
	}

	std::stringstream resultss;
	resultss << datastring << resultfname;
	std::ofstream resultfile(resultss.str());
	if (!resultfile.is_open()) {
		std::cout << "Couldn't open result file" << std::endl;
		system("pause");
		return 0;
	}

	std::ifstream intervalfile;
	if (usingIntervalFile) {
		std::stringstream intss;
		intss << datastring << intervalfname;
		intervalfile.open(intss.str());
		if (!intervalfile.is_open()) {
			std::cout << "Couldn't open interval file" << std::endl;
			system("pause");
			return 0;
		}
	}

	size_t inum = 1;
	while (!usingIntervalFile || readIntervalParameters(&intervalfile)) {
		std::stringstream datass;
		datass << datastring << datafname;
		std::ifstream datafile(datass.str());
		if (!datafile.is_open()) {
			std::cout << "Couldn't open data file" << std::endl;
			system("pause");
			return 0;
		}

		std::vector<std::vector<std::vector<float>>> intervalData;
		intervalData.resize(columns.size());
		inputs.resize(columns.size());
		for (size_t i = 0; i < columns.size(); i++) {
			intervalData[i].clear();
			inputs[i].clear();
		}
		size_t iBegin;
		size_t iEnd;
		currentLineNum = 1;
		while (readIntervalData(&datafile, &intervalData, &iBegin, &iEnd)) {
			if (print)
				std::cout << "Interval #" << inum << " (" << datafname << " " << iBegin << "-" << iEnd << ") ";
			if (keepExtraData) {
				std::cout << "Extra data: ";
				for (size_t i = 0; i < extraData.size(); i++) {
					std::cout << extraData[i] << " ";
				}
			}
			if (print)
				std::cout << " | " << std::endl;
			resultfile << inum << " " << datafname << " " << iBegin << " " << iEnd << " ";
			saveIntervalResult(&resultfile, &intervalData, print);
			inum++;
		}
		if (!usingIntervalFile)
			break;
	}

	std::cout << "Done." << std::endl;
	system("pause");
}

void createWeightLayers() {
	weightlayers.push_back((Layer*)new ConvolutionLayer(1, 136, NUM_NEURONS, 128, 9, TRANSFER_TYPE_RECTIFIER));
	weightlayers.push_back((Layer*)new MaxPoolLayer(NUM_NEURONS, 128));
	weightlayers.push_back((Layer*)new ConvolutionLayer(NUM_NEURONS, 64, NUM_NEURONS, 60, 5, TRANSFER_TYPE_RECTIFIER));
	weightlayers.push_back((Layer*)new MaxPoolLayer(NUM_NEURONS, 60));
	weightlayers.push_back((Layer*)new ConvolutionLayer(NUM_NEURONS, 30, NUM_NEURONS, 26, 5, TRANSFER_TYPE_RECTIFIER));
	weightlayers.push_back((Layer*)new MaxPoolLayer(NUM_NEURONS, 26));
	weightlayers.push_back((Layer*)new ConvolutionLayer(NUM_NEURONS, 13, NUM_NEURONS, 10, 4, TRANSFER_TYPE_RECTIFIER));
	weightlayers.push_back((Layer*)new MaxPoolLayer(NUM_NEURONS, 10));
	weightlayers.push_back((Layer*)new ConvolutionLayer(NUM_NEURONS, 5, NUM_NEURONS, 4, 2, TRANSFER_TYPE_RECTIFIER));
	weightlayers.push_back((Layer*)new MaxPoolLayer(NUM_NEURONS, 4));
	weightlayers.push_back((Layer*)new FixedLayer(2 * NUM_NEURONS, numFixedHiddenNeurons, TRANSFER_TYPE_RECTIFIER));
	weightlayers.push_back((Layer*)new FixedLayer(numFixedHiddenNeurons, 1, TRANSFER_TYPE_LINEAR));

	for (size_t i = 1; i < weightlayers.size(); i++) {
		weightlayers[i]->link(weightlayers[i - 1]);
	}
}

bool loadWeights(std::vector<Layer*> layers, std::string fname) {
	std::stringstream fss;
	fss << savestring << fname;

	std::ifstream infile(fss.str().c_str());

	if (!infile.is_open()) {
		std::cout << "Couldn't open weights file" << std::endl;
		throw new std::runtime_error("");
	}

	for (size_t lay = 0; lay < layers.size(); lay++) {
		layers[lay]->loadWeights(&infile);
	}

	return true;
}

void loadParameters(std::string parName) {
	std::ifstream infile(parName.c_str());
	std::string line;
	while (getline(infile, line)) {
		std::stringstream lss(line);
		std::string var;
		lss >> var;
		if (var == "NUM_NEURONS")
			lss >> NUM_NEURONS;
		else if (var == "savename")
			lss >> savename;
		else if (var == "numFixedHiddenNeurons")
			lss >> numFixedHiddenNeurons;
		else if (var == "NUM_INPUTS") {
			size_t numInputs;
			lss >> numInputs;
			if (numInputs != 136) {
				std::cout << "Currently only supports NUM_INPUTS=136" << std::endl;
				throw new std::runtime_error("");
			}
		}
	}
}

bool readIntervalData(std::ifstream* datafile, std::vector<std::vector<std::vector<float>>>* intervalData, size_t* iBegin, size_t* iEnd) {
	for (size_t i = 0; i < intervalData->size();i++)
		(*intervalData)[i].clear();
	std::string line;

	for (; currentLineNum < begin && std::getline((*datafile), line); currentLineNum++) {}

	*iBegin = currentLineNum;

	size_t i;
	for (i = 0; i < intervalSize && std::getline((*datafile), line); i++) {
		if (end > 0 && currentLineNum > end) {
			if (!discardIntervalRemainder)
				break;
			else
				return false;
		}
		currentLineNum++;
		std::string dum;
		std::stringstream liness(line);
		float in;
		size_t prevColumn = 1;
		for (size_t c = 0; c < columns.size(); c++) {
			for (size_t j=prevColumn; j < columns[c]; j++) {
				liness >> dum;
			}
			prevColumn = columns[c] + 1;
			liness >> in;
			inputs[c].push_back(in);
			if (inputs[c].size() > NUM_INPUTS)
				inputs[c].pop_front();

			if (inputs[c].size() == NUM_INPUTS) {
				std::vector<float> io;
				io.resize(NUM_INPUTS);
				size_t n = 0;
				for (std::list<float>::iterator it = inputs[c].begin(); it != inputs[c].end(); it++) {
					io[n] = *it;
					n++;
				}

				float maxinput = -999999;
				float mininput = 999999;
				for (size_t j = 0; j < NUM_INPUTS; j++) {
					if (io[j] > maxinput)
						maxinput = io[j];
					if (io[j] < mininput)
						mininput = io[j];
				}
				for (size_t j = 0; j<NUM_INPUTS; j++) {
					if (maxinput > mininput)
						io[j] = 2 * (io[j] - mininput) / (maxinput - mininput) - 1;
					else
						io[j] = 0;
				}

				if (!discard || !discardInput(&io[0]))
					(*intervalData)[c].push_back(io);
			}
		}
	}
	*iEnd = currentLineNum - 1;
	return i > 0;
}

bool discardInput(float* inputs) {
	if (NUM_INPUTS < 10)
		return false;

	float begAvg = 0;
	for (size_t i = 0; i < 5; i++) {
		begAvg += inputs[i];
	}
	begAvg /= 5;

	float endAvg = 0;
	for (size_t i = 0; i < 5; i++) {
		endAvg += inputs[NUM_INPUTS - i - 1];
	}
	endAvg /= 5;

	return fabs(begAvg - endAvg) > 1;
}

void saveIntervalResult(std::ofstream* resultfile, std::vector<std::vector<std::vector<float>>>* intervalData, bool print) {
	size_t columnnum = 1;
	for (size_t c = 0; c < columns.size(); c++) {
		std::vector<float> sampleoutputs;
	
		for (size_t i = 0; i < (*intervalData)[c].size(); i++) {
			//----calculate----
			memcpy(&weightlayers[0]->inlayer[0], &(*intervalData)[c][i][0], NUM_INPUTS*sizeof(float));

			for (size_t l = 0; l < weightlayers.size(); l++) {
				weightlayers[l]->calc();
			}

			//----end calculate-----

#ifdef DEBUG_SAVE_INTERNAL_RESULTS
			debugSaveInternalResults();
#endif
			float* outlayer = weightlayers[weightlayers.size() - 1]->outlayer;
			sampleoutputs.push_back(outlayer[0]);
		}

		float origmean = mean(sampleoutputs);
		float origdev = stdev(sampleoutputs, origmean);

		size_t numOutliersDiscarded = 0;
		size_t numTotalPoints = sampleoutputs.size();
		if (origdev > 0.1f) {
			for (size_t i = 0; i < sampleoutputs.size(); i++) {
				if (fabs(sampleoutputs[i] - origmean) > origdev) {
					sampleoutputs.erase(sampleoutputs.begin() + i);
					i--;
					numOutliersDiscarded++;
				}
			}
		}

		float newmean = mean(sampleoutputs);
		float newstdev = stdev(sampleoutputs, newmean);

		if (print)
			std::cout << "    Column " << columns[c] << ": " << newmean << "+/-" << newstdev << "(" << numOutliersDiscarded << "/" << numTotalPoints << " outliers discarded)" << std::endl;

		while (!usingIntervalFile && columnnum < columns[c]) {
			(*resultfile) << "0 0 ";
			columnnum++;
		}

		if (usingIntervalFile)
			(*resultfile) << columns[c] << " ";

		(*resultfile) << newmean << " " << newstdev << " " << numTotalPoints-numOutliersDiscarded << " " << numTotalPoints << " ";
		columnnum++;
	}
	if (keepExtraData) {
		for (size_t i = 0; i < extraData.size(); i++) {
			(*resultfile) << extraData[i] << " ";
		}
	}
	(*resultfile) << std::endl;
}

bool readIntervalParameters(std::ifstream* intervalfile) {
	std::string line;
	bool done = true;
	if (getline((*intervalfile), line)) done = false;
	if (!done) {
		std::stringstream lss(line);
		size_t column;
		lss >> datafname >> column >> begin >> end;
		columns.clear();
		columns.push_back(column);
		extraData.clear();
		if (keepExtraData) {
			std::string dum;
			while (lss >> dum)
				extraData.push_back(dum);
		}
	}
	return !done;
}

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
