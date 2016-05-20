#include "stockrun.cuh"
#include "Shlwapi.h"

#define nIter 5

#define initialTrainSamples 2
#define trainIncreaseThreshold 100.0f

#define initialStepMult 10.0f
//float initialStepMult = 40.0f;
#define minimumStepMult 1.0f
//float minimumStepMult = 0.5f;
#define stepMultDecFactor 0.707f
//float stepMultDecFactor = 0.75f;

#define datastring "rawdata/"
#define savestring "saveweights/"

#define RAND_EXPLICIT			//uses a test set with inputs in a random order (specified by randtrainstring)

void saveResults(size_t numRuns, float afterError);
void saveSetHistory(size_t nSamples, size_t nRuns, float stepFacMult);
void loadLocalParameters();
void loadSimVariables();
void saveSimVariables();
size_t readData(size_t begin, size_t numIOs);

size_t numRuns;
size_t trainSamples;
size_t totalSamples;
size_t numRunSetStart;

float stepMult;

float stepMultiplier(size_t numRuns) {
	return stepMult;
}

int main() {
	srand((size_t)time(NULL));
	loadParameters();
	loadLocalParameters();

	setStrings(datastring, savestring);
	numRuns = 0;
	trainSamples = initialTrainSamples;
	stepMult = initialStepMult;
	numRunSetStart = 0;

	loadSimVariables();

	LayerCollection layers = createLayerCollection();
	initializeLayers(&layers);
	loadWeights(layers, savename);

	size_t numSamples = readData(1, trainSamples);

	while (true) {
		std::cout << nIter << "+1 runs on " << numSamples << " samples: ";
		auto gpustart = std::chrono::high_resolution_clock::now();

		randomizeTrainSet();

		for (size_t i = 0; i < nIter; i++) {
			runSim(layers, true, stepMultiplier(numRuns));
		}

		float afterError = runSim(layers, false, 0);

		auto gpuelapsed = std::chrono::high_resolution_clock::now() - gpustart;
		long long gputime = std::chrono::duration_cast<std::chrono::microseconds>(gpuelapsed).count();
		saveWeights(layers, savename);
		numRuns += nIter;
		std::cout << gputime/1000000 << " s, Error: " << afterError << std::endl;

		if (afterError < trainIncreaseThreshold && trainSamples < totalSamples) {
			saveSetHistory(numSamples, numRuns - numRunSetStart, stepMult);
			numRunSetStart = numRuns;
			trainSamples = min(2 * trainSamples, totalSamples);
			numSamples = readData(1, trainSamples);
			if (stepMultDecFactor > 0)
				stepMult = max(stepMultDecFactor*stepMult, minimumStepMult);
		}

		saveResults(numRuns, afterError);
		saveSimVariables();
	}
}

void saveResults(size_t numRuns, float afterError) {
	std::stringstream resname;
	resname << savestring << savename << "result";
	std::ofstream resfile(resname.str().c_str(), std::ios_base::app);
	resfile << numRuns << " " << afterError << std::endl;
}

void saveSetHistory(size_t nSamples, size_t nRuns, float stepFacMult) {
	std::stringstream hisname;
	hisname << savestring << savename << "history";
	std::ofstream hisfile(hisname.str().c_str(), std::ios_base::app);
	hisfile << nSamples << " " << stepFacMult << " " << nRuns << std::endl;
}

void loadSimVariables() {
	std::stringstream pss;
	pss << savestring << savename << "pars";
	std::ifstream infile(pss.str().c_str());

	std::string line;
	while (getline(infile, line)) {
		std::stringstream lss(line);
		std::string var;
		lss >> var;
		if (var == "numRuns")
			lss >> numRuns;
		if (var == "trainSamples")
			lss >> trainSamples;
		if (var == "stepMult")
			lss >> stepMult;
		if (var == "numRunSetStart")
			lss >> numRunSetStart;
	}
}

void saveSimVariables() {
	std::stringstream pss;
	pss << savestring << savename << "pars";
	std::ofstream outfile(pss.str().c_str());

	outfile << "numRuns " << numRuns << std::endl;
	outfile << "trainSamples " << trainSamples << std::endl;
	outfile << "stepMult " << stepMult << std::endl;
	outfile << "numRunSetStart " << numRunSetStart << std::endl;
}

size_t readData(size_t begin, size_t numIOs) {
	std::cout << "Reading " << numIOs << " samples from trainset: ";
	auto readstart = std::chrono::high_resolution_clock::now();
#ifndef RAND_EXPLICIT
	totalSamples = readTrainSet(trainstring, begin, numIOs);
#else
	totalSamples = readExplicitTrainSet(randtrainstring, begin, numIOs);
#endif
	auto readelapsed = std::chrono::high_resolution_clock::now() - readstart;
	long long readtime = std::chrono::duration_cast<std::chrono::microseconds>(readelapsed).count();
	std::cout << readtime / 1000000 << " s" << std::endl;
	size_t numSamples = min(numIOs, totalSamples);
	std::cout << numSamples << "/" << totalSamples << " samples loaded" << std::endl;
	return numSamples;
}

void loadLocalParameters() {
	/*
	std::ifstream infile("pars.cfg");
	std::string line;
	while (getline(infile, line)) {
		std::stringstream lss(line);
		std::string var;
		lss >> var;

		if (var == "initialStepMult")
			lss >> initialStepMult;
		else if (var == "minimumStepMult")
			lss >> minimumStepMult;
		else if (var == "stepMultDecFactor")
			lss >> stepMultDecFactor;
	}
	*/
}
