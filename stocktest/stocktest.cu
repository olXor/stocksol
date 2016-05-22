#include <stockrun.cuh>
#include "Shlwapi.h"

#define LOCAL

#ifdef LOCAL
#define datastring "rawdata/"
#define savestring "saveweights/"
#else
#define datastring "../stockproj/rawdata/"
#define savestring "../stockproj/saveweights/"
#endif

//#define RAND_EXPLICIT
size_t testBegin = 1;
size_t testNumIOs = 0;

bool testUseSampleFile = false;
bool discardSamples = false;
std::string testOutputFile = "testoutput";

size_t readData(size_t begin, size_t numIOs);
void loadLocalParameters(std::string parName);

int main() {
	srand((size_t)time(NULL));
#ifdef LOCAL
	loadParameters("pars.cfg");
	loadLocalParameters("pars.cfg");
#else
	loadLocalParameters("../stockproj/pars.cfg");
	loadParameters("../stockproj/pars.cfg");
#endif
	setStrings(datastring, savestring);

	LayerCollection layers = createLayerCollection();
	initializeLayers(&layers);
	loadWeights(layers, savename);

	readData(testBegin, testNumIOs);

	if (testUseSampleFile)
		sampleTestSim(layers, testOutputFile);
	else
		runSim(layers, false, 0, true);

#ifdef LOCAL
	system("pause");
#endif
}

size_t readData(size_t begin, size_t numIOs) {
	size_t numSamples;
	if (!testUseSampleFile) {
		std::cout << "Reading " << numIOs << " samples from trainset: ";
		auto readstart = std::chrono::high_resolution_clock::now();
#ifndef RAND_EXPLICIT
		size_t totalSamples = readTrainSet(trainstring, begin, numIOs);
#else
		size_t totalSamples = readExplicitTrainSet(randtrainstring, begin, numIOs);
#endif
		auto readelapsed = std::chrono::high_resolution_clock::now() - readstart;
		long long readtime = std::chrono::duration_cast<std::chrono::microseconds>(readelapsed).count();
		std::cout << readtime / 1000000 << " s" << std::endl;
		numSamples = min(numIOs, totalSamples);
		std::cout << numSamples << "/" << totalSamples << " samples loaded" << std::endl;
	}
	else {
		std::cout << "Reading trainset: ";
		auto readstart = std::chrono::high_resolution_clock::now();
		size_t numDiscards[2];

		sampleReadTrainSet(trainstring, discardSamples, numDiscards);

		auto readelapsed = std::chrono::high_resolution_clock::now() - readstart;
		long long readtime = std::chrono::duration_cast<std::chrono::microseconds>(readelapsed).count();
		std::cout << readtime / 1000000 << " s" << std::endl;

		std::cout << numDiscards[0] << "/" << numDiscards[1] << " samples discarded" << std::endl;

		numSamples = numDiscards[1] - numDiscards[0];
	}
	return numSamples;
}

void loadLocalParameters(std::string parName) {
	std::ifstream infile(parName.c_str());
	std::string line;
	while (getline(infile, line)) {
		std::stringstream lss(line);
		std::string var;
		lss >> var;
		if (var == "testUseSampleFile")
			lss >> testUseSampleFile;
		else if (var == "testBegin")
			lss >> testBegin;
		else if (var == "testNumIOs")
			lss >> testNumIOs;
		else if (var == "discardSamples")
			lss >> discardSamples;
		else if (var == "testOutputFile")
			lss >> testOutputFile;
	}
}
