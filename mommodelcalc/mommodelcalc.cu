#include <stockrun.cuh>
#include "Shlwapi.h"

#ifdef LOCAL
#define datastring "rawdata/"
#define savestring "saveweights/"
#else
#define datastring "../stockproj/rawdata/"
#define savestring "../stockproj/saveweights/"
#endif

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

LayerCollection weightlayers;

std::vector<std::list<float>> inputs;

bool readIntervalParameters(std::ifstream* intervalfile);
bool readIntervalData(std::ifstream* datafile, std::vector<std::vector<IOPair>>* intervalData, size_t* iBegin, size_t* iEnd);
void saveIntervalResult(std::ofstream* resultfile, std::vector<std::vector<IOPair>>* intervalData, bool print);

int main() {
	srand((size_t)time(NULL));

#ifdef LOCAL
	loadParameters("pars.cfg");
#else
	loadParameters("../stockproj/pars.cfg");
#endif
	setStrings(datastring, savestring);

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

	weightlayers = createLayerCollection(0, FULL_NETWORK);
	initializeLayers(&weightlayers);
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

		std::vector<std::vector<IOPair>> intervalData;
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

bool readIntervalData(std::ifstream* datafile, std::vector<std::vector<IOPair>>* intervalData, size_t* iBegin, size_t* iEnd) {
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
				IOPair io;
				io.inputs.resize(NUM_INPUTS);
				size_t n = 0;
				for (std::list<float>::iterator it = inputs[c].begin(); it != inputs[c].end(); it++) {
					io.inputs[n] = *it;
					n++;
				}

				float maxinput = -999999;
				float mininput = 999999;
				for (size_t j = 0; j < NUM_INPUTS; j++) {
					if (io.inputs[j] > maxinput)
						maxinput = io.inputs[j];
					if (io.inputs[j] < mininput)
						mininput = io.inputs[j];
				}
				for (size_t j = 0; j<NUM_INPUTS; j++) {
					if (maxinput > mininput)
						io.inputs[j] = 2 * (io.inputs[j] - mininput) / (maxinput - mininput) - 1;
					else
						io.inputs[j] = 0;
				}

				if (!discard || !discardInput(&io.inputs[0]))
					(*intervalData)[c].push_back(io);
			}
		}
	}
	*iEnd = currentLineNum - 1;
	return i > 0;
}

void saveIntervalResult(std::ofstream* resultfile, std::vector<std::vector<IOPair>>* intervalData, bool print) {
	float* d_inputs;
	if (weightlayers.numConvolutions > 0) {
		if (weightlayers.convPars[0].numInputLocs != NUM_INPUTS || weightlayers.convPars[0].numInputNeurons != 1)
			throw std::runtime_error("inputs to first layer don't match data set");
		d_inputs = weightlayers.convMat[0].inlayer;
	}
	else if (weightlayers.numFixedNets > 0) {
		if (weightlayers.fixedPars[0].numInputNeurons != NUM_INPUTS)
			throw std::runtime_error("inputs to first layer don't match data set");
		d_inputs = weightlayers.fixedMat[0].inlayer;
	}
	else
		throw std::runtime_error("tried to run on a network with no convolutions and no fixed networks");

	float* h_output = new float[numBins];

	if (numBins > 1) {
		std::cout << "This program doesn't work with binned networks at the moment, sorry!" << std::endl;
		throw new std::runtime_error("This program doesn't work with binned networks at the moment, sorry!");
	}

	disableDropout();
	generateDropoutMask(&weightlayers);

	size_t columnnum = 1;
	for (size_t c = 0; c < columns.size(); c++) {
		std::vector<float> sampleoutputs;
	
		for (size_t i = 0; i < (*intervalData)[c].size(); i++) {
			//----calculate----
			checkCudaErrors(cudaMemcpy(d_inputs, &(*intervalData)[c][i].inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice));

			calculate(weightlayers);

			checkCudaErrors(cudaMemcpy(h_output, weightlayers.fixedMat[weightlayers.numFixedNets - 1].outlayer, numBins*sizeof(float), cudaMemcpyDeviceToHost));
			//----end calculate-----

			sampleoutputs.push_back(h_output[0]);
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

		(*resultfile) << newmean << " " << newstdev << " ";
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
