#include <stockrun.cuh>
#include "Shlwapi.h"

#define datastring "../stockproj/rawdata/"
#define savestring "../stockproj/saveweights/"

#define RAND_EXPLICIT
#define testBegin 1
#define testEnd 16

void loadParameters();
size_t readData(size_t begin, size_t numIOs);

int main() {
	srand((size_t)time(NULL));
	loadParameters("../stockproj/pars.cfg");
	setStrings(datastring, savestring);

	LayerCollection layers = createLayerCollection();
	initializeLayers(&layers);
	loadWeights(layers, savename);

	readData(testBegin, testEnd);

	runSim(layers, false, 0, true);
}

size_t readData(size_t begin, size_t numIOs) {
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
	size_t numSamples = min(numIOs, totalSamples);
	std::cout << numSamples << "/" << totalSamples << " samples loaded" << std::endl;
	return numSamples;
}
