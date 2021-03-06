#include "stockrun.cuh"
#include "Shlwapi.h"

size_t nIter = 5;

size_t initialTrainSamples = 2;
size_t trainSamplesIncreaseFactor = 2;
float trainIncreaseThreshold = 100.0f;

float initialStepMult = 40.0f;
float minimumStepMult = 0.5f;
float stepMultDecFactor = 0.707f;
float annealingStartError = 500.0f;

size_t backupInterval = 0;
size_t backupSampleNumStart = 0;

#define datastring "rawdata/"
#define savestring "saveweights/"

#define RAND_EXPLICIT			//uses a test set with inputs in a random order (specified by randtrainstring)

void saveResults(size_t numRuns, float afterError, float* afterSecError, float testAfterError, float* testAfterSecError, float testSampleAfterError);
bool loadLastResults(float* afterError, float* afterSecError, float* testAfterError, float* testAfterSecError, float* testSampleAfterError);
void saveSetHistory(size_t nSamples, size_t nRuns, float stepFacMult);
void loadLocalParameters();
void loadSimVariables(bool userParams = false);
void saveSimVariables();
size_t readData(size_t begin, size_t numIOs, bool readTestSet = false);
void backupFiles(std::string backname);

size_t numRuns;
size_t trainSamples;
size_t totalSamples;
size_t numRunSetStart;

float stepMult;
float stepAdjustment = 1.0f;
size_t stepAdjustmentNumStart = 0.0f;

float redoErrorThreshold = 0.0f;
float redoStepAdjustment = 1.0f;
float successStepAdjustment = 1.0f;

bool randomizeTrainSetEveryRun = true;
bool randomizeSubsetOnThreshold = false;

bool pairedTraining = false;

size_t primaryErrorType = 0;
size_t testMinErrorType = 0;

#define ERRORS_SAVED 5
std::list<float> lastErrors;

bool printTestError = false;
size_t minTrainSizeToPrintTestError = 0;
bool testExplicitFile = false;
std::string testfile = "testset";
bool testUseSampleFile = true;
bool discardSamples = false;

#define numSecErrors 5

bool backupMinTestError = true;
float minTestError = 9999999.0f;
bool backupMinTestErrorSample = true;

bool trainStockBAGSet = false;
size_t trainBAGSubnetNum = 1;
bool addBagTradeSuffix = false;
bool appendBAGSubnetNum = true;

size_t numRunsOnFullTrainset = 0;
float fullTrainsetErrorGoal = 0.0f;
size_t numRunsToStallRestart = 0;
float stallRestartThreshold = 0.0f;
float minTestErrorStallRestartThreshold = 0.0f;

size_t numTrainCrossValSets = 0;

void moveSaveToStallBackup();

size_t resultSaveSampleNumStart = 0;

size_t declaredTotalSamples = 0;

float binOutAverageTrainReg = 0.0f;

bool multTrainSetsPerCV = true;

float fullTrainSetStepAdjustment = 0;

float annealingMultiplier() {
	if (lastErrors.size() == 0 || annealingStartError == 0)
		return 1;

	float avg = 0;
	for (std::list<float>::const_iterator it = lastErrors.begin(); it != lastErrors.end(); it++) {
		avg += *it;
	}
	avg /= lastErrors.size();

	if (avg > annealingStartError)
		return 1;
	return avg / annealingStartError;
}

float stepMultiplier(size_t numRuns) {
	return annealingMultiplier()*stepMult*stepAdjustment;
}

void updateLastErrors(float error) {
	lastErrors.push_back(error);
	if (lastErrors.size() > ERRORS_SAVED)
		lastErrors.pop_front();
}

int main() {
	srand((size_t)time(NULL));

	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
	if (!prop.canMapHostMemory)
		exit(0);
	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

	setStrings(datastring, savestring);

	loadParameters("pars.cfg");
	loadLocalParameters();

	if (binOutAverageTrainReg != 0)
		initializeDOutAverages();

	size_t numBagSubnets = 1;
	if (trainStockBAGSet) {
		if (addBagTradeSuffix)
			numBagSubnets = 2 * trainBAGSubnetNum;
		else
			numBagSubnets = trainBAGSubnetNum;
	}
	bool skipDataSetLoad = false;
	//break out of the loop after one iteration if trainStockBAGSet = false
	for (size_t bagNum = 0; bagNum < numBagSubnets; bagNum++) {
		for (size_t cv = 0; cv < numTrainCrossValSets || numTrainCrossValSets == 0; cv++) {
			stepAdjustment = 1.0f;
			loadParameters("pars.cfg");
			loadLocalParameters();
			if (trainStockBAGSet) {
				std::stringstream suff1ss;
				std::stringstream suff2ss;
				if (addBagTradeSuffix) {
					if (bagNum % 2 == 0)
						suff1ss << "long";
					else
						suff1ss << "short";
				}
				if (numTrainCrossValSets != 0)
					suff1ss << cv + 1;
				if (appendBAGSubnetNum) {
					if (numTrainCrossValSets != 0)
						suff2ss << "-";
					if (addBagTradeSuffix)
						suff2ss << bagNum / 2 + 1;
					else
						suff2ss << bagNum + 1;
				}
				std::stringstream suff;
				if (multTrainSetsPerCV)
					suff << suff1ss.str() << suff2ss.str();
				else
					suff << suff1ss.str();
				std::stringstream savess;
				savess << savename << suff1ss.str() << suff2ss.str();
				savename = savess.str();
				std::stringstream rantrainss;
				rantrainss << randtrainstring << suff.str();
				randtrainstring = rantrainss.str();
				std::stringstream testss;
				testss << testfile << suff.str();
				testfile = testss.str();
				std::stringstream binss;
				binss << binEdgeFile << suff.str();
				loadBinEdges(binss.str());
				std::cout << std::endl << "Beginning run on " << savename << std::endl;
			}
			else {
				loadBinEdges(binEdgeFile);
			}
			numRuns = 0;
			trainSamples = initialTrainSamples;
			stepMult = initialStepMult;
			numRunSetStart = 0;
			lastErrors.clear();
			minTestError = 9999999.0f;

			loadSimVariables();

			float initError;
			float* initSecError = new float[numSecErrors];
			for (size_t i = 0; i < numSecErrors; i++)
				initSecError[i] = 0.0f;
			float testInitError;
			float* testInitSecError = new float[numSecErrors];
			for (size_t i = 0; i < numSecErrors; i++)
				testInitSecError[i] = 0.0f;
			float testSampleInitError;

			if (loadLastResults(&initError, initSecError, &testInitError, testInitSecError, &testSampleInitError)) {
				std::cout << "Loading previous results file:";
				float prevPrimaryAfterError;
				if (primaryErrorType == 0)
					prevPrimaryAfterError = initError;
				else
					prevPrimaryAfterError = initSecError[primaryErrorType - 1];

				std::cout << " Last result " << prevPrimaryAfterError << "/" << fullTrainsetErrorGoal << std::endl;

				if (trainSamples >= declaredTotalSamples && ((numRunsOnFullTrainset != 0 && numRuns - numRunSetStart >= numRunsOnFullTrainset) || (prevPrimaryAfterError <= fullTrainsetErrorGoal))) {
					std::cout << "Run completed after " << numRuns - numRunSetStart << " rounds on full trainset" << std::endl;
					continue;
				}
			}
			else
				std::cout << "No previous results found" << std::endl;

			LayerCollection layers;
			PairedConvCollection pairedLayers;
			if (pairedTraining) {
				pairedLayers = createAndInitializePairedConvCollection(NUM_INPUTS);
				if (!loadPairedWeights(pairedLayers, savename))
					savePairedWeights(pairedLayers, savename);
			}
			else {
				layers = createLayerCollection(0, getLCType());
				initializeLayers(&layers);
				if (!loadWeights(layers, savename))
					saveWeights(layers, savename);
			}

			size_t numSamples;
			if (!skipDataSetLoad) {
				if (randomizeSubsetOnThreshold) {
					size_t tSamples = readData(1, 0, printTestError);
					randomizeTrainSet();
					if (trainSamples != 0)
						numSamples = min(trainSamples, printTestError);
					else
						numSamples = tSamples;
				}
				else {
					numSamples = readData(1, trainSamples, true);
				}
				trainSamples = numSamples;
			}
			skipDataSetLoad = false;

			fillTrainsetIndicesByBin();
			generateTrainWeightBins();
			loadSimVariables();	//not really efficent to do this again but whatever

			std::cout << "Calculating initial error: ";
			disableDropout();
			auto initstart = std::chrono::high_resolution_clock::now();
			if (pairedTraining)
				initError = runPairedSim(pairedLayers, false, 0, trainSamples);
			else {
				initError = runSim(layers, false, 0, trainSamples, false, initSecError);
			}
			if (printTestError && !pairedTraining) {
				testInitError = runSim(layers, false, 0, 0, false, testInitSecError, true);
				if (testUseSampleFile)
					testSampleInitError = sampleTestSim(layers, NULL, false, false, true);
			}

			auto initelapsed = std::chrono::high_resolution_clock::now() - initstart;
			long long inittime = std::chrono::duration_cast<std::chrono::microseconds>(initelapsed).count();

			std::cout << inittime / 1000000 << " s, Error: " << initError;
			for (size_t i = 0; i < numSecErrors; i++) {
				if (initSecError[i] != 0.0f)
					std::cout << " SE" << i + 1 << ": " << initSecError[i];
			}
			if (printTestError && !pairedTraining) {
				std::cout << " | Test Error: " << testInitError;
				for (size_t i = 0; i < numSecErrors; i++) {
					if (testInitSecError[i] != 0.0f)
						std::cout << " SE" << i + 1 << ": " << testInitSecError[i];
				}
				if (testUseSampleFile)
					std::cout << " | Sample Test Error: " << testSampleInitError;
			}
			std::cout << std::endl;

			float prevPrimaryAfterError;
			if (primaryErrorType == 0)
				prevPrimaryAfterError = initError;
			else
				prevPrimaryAfterError = initSecError[primaryErrorType - 1];
			updateLastErrors(prevPrimaryAfterError);

			delete[] initSecError;
			delete[] testInitSecError;

			bool belowStallThresh = false;
			std::vector<float> outAverages(numBins);
			std::vector<float> lastGoodOutAverages(numBins);
			while (true) {
				if (pairedTraining)
					std::cout << nIter << "+1 runs on " << numSamples << " sample pairs";
				else
					std::cout << nIter << "+1 runs on " << numSamples << " samples";
				if (trainSamples >= declaredTotalSamples && fullTrainSetStepAdjustment > 0)
					stepAdjustment = fullTrainSetStepAdjustment;
				if (trainSamples >= stepAdjustmentNumStart && redoErrorThreshold > 0.0f) {
					std::cout << "(SA: " << stepAdjustment << ")";
				}
				std::cout << ": ";
				auto gpustart = std::chrono::high_resolution_clock::now();

				if (randomizeSubsetOnThreshold) {
					randomizeTrainSet(trainSamples);
				}
				else if (randomizeTrainSetEveryRun)
					randomizeTrainSet();

				enableDropout();
				for (size_t i = 0; i < nIter; i++) {
					if (pairedTraining)
						runPairedSim(pairedLayers, true, stepMultiplier(numRuns), trainSamples);
					else {
						if (binOutAverageTrainReg != 0)
							runSim(layers, true, stepMultiplier(numRuns), trainSamples, false, NULL, false, &outAverages[0], NULL, false);
						else
							runSim(layers, true, stepMultiplier(numRuns), trainSamples, false, NULL, false, NULL, NULL, false);
					}
					if (binOutAverageTrainReg != 0) {
						for(size_t j=0;j<numBins;j++) {
							outAverages[j] *= binOutAverageTrainReg;
						}
					}
#ifdef BATCH_MODE
					if (pairedTraining) {
						batchUpdate(pairedLayers.conv1);
						batchUpdate(pairedLayers.conv2);
						batchUpdate(pairedLayers.fixed);
					}
					else
						batchUpdate(layers);
#endif
				}


				float afterError;
				float* afterSecError = new float[numSecErrors];
				for (size_t i = 0; i < numSecErrors; i++)
					afterSecError[i] = 0.0f;
				disableDropout();
				if (pairedTraining)
					afterError = runPairedSim(pairedLayers, false, 0, trainSamples);
				else {
					afterError = runSim(layers, false, 0, trainSamples, false, afterSecError);
				}

				float testAfterError = 0.0f;
				float* testAfterSecError = new float[numSecErrors];
				for (size_t i = 0; i < numSecErrors; i++)
					testAfterSecError[i] = 0.0f;
				float testSampleAfterError;
				if (printTestError && trainSamples >= minTrainSizeToPrintTestError && !pairedTraining) {
					for (size_t i = 0; i < numSecErrors; i++)
						testAfterSecError[i] = 0.0f;
					testAfterError = runSim(layers, false, 0, 0, false, testAfterSecError, true, NULL, NULL, true);
					if (testUseSampleFile)
						testSampleAfterError = sampleTestSim(layers, NULL, false, false, true);
				}

				auto gpuelapsed = std::chrono::high_resolution_clock::now() - gpustart;
				long long gputime = std::chrono::duration_cast<std::chrono::microseconds>(gpuelapsed).count();
				std::cout << gputime / 1000000 << " s, Error: " << afterError;
				for (size_t i = 0; i < numSecErrors; i++) {
					if (afterSecError[i] != 0.0f)
						std::cout << " SE" << i + 1 << ": " << afterSecError[i];
				}
				if (printTestError && trainSamples >= minTrainSizeToPrintTestError && !pairedTraining) {
					std::cout << " | Test Error: " << testAfterError;
					for (size_t i = 0; i < numSecErrors; i++) {
						if (testAfterSecError[i] != 0.0f)
							std::cout << " SE" << i + 1 << ": " << testAfterSecError[i];
					}

					if (testUseSampleFile)
						std::cout << " | Test Sample Error: " << testSampleAfterError;
				}
				std::cout << std::endl;

				loadSimVariables(true);	//to allow the user to change variables mid-run

				float primaryAfterError;
				if (primaryErrorType == 0)
					primaryAfterError = afterError;
				else
					primaryAfterError = afterSecError[primaryErrorType - 1];

				if (trainSamples >= stepAdjustmentNumStart) {
					if (redoErrorThreshold > 0.0f && prevPrimaryAfterError > 0.0f && primaryAfterError - prevPrimaryAfterError > redoErrorThreshold) {
						std::cout << "Error increase was above threshold; redoing last round" << std::endl;

						if (pairedTraining)
							loadPairedWeights(pairedLayers, savename);
						else
							loadWeights(layers, savename);
						loadSimVariables();
						stepAdjustment *= redoStepAdjustment;
						saveSimVariables();
						outAverages = lastGoodOutAverages;
						continue;
					}
					stepAdjustment *= successStepAdjustment;
					lastGoodOutAverages = outAverages;
				}
				prevPrimaryAfterError = primaryAfterError;
				updateLastErrors(primaryAfterError);
				if (pairedTraining)
					savePairedWeights(pairedLayers, savename);
				else
					saveWeights(layers, savename);
				numRuns += nIter;

				if (trainSamples >= resultSaveSampleNumStart)
					saveResults(numRuns, afterError, afterSecError, testAfterError, testAfterSecError, testSampleAfterError);

				if (trainSamples >= backupSampleNumStart && backupInterval > 0 && (numRuns - numRunSetStart) % backupInterval == 0) {
					std::stringstream bss;
					bss << savename << numSamples << "-" << numRuns - numRunSetStart;
					backupFiles(bss.str().c_str());
				}

				float testMinAfterError;
				if (testUseSampleFile && backupMinTestErrorSample)
					testMinAfterError = testSampleAfterError;
				else {
					if (testMinErrorType == 0)
						testMinAfterError = testAfterError;
					else
						testMinAfterError = testAfterSecError[testMinErrorType - 1];
				}

				if (backupMinTestError && testMinAfterError < minTestError && printTestError && trainSamples >= minTrainSizeToPrintTestError) {
					minTestError = testMinAfterError;
					saveSimVariables();
					std::stringstream bss;
					bss << savename << "Min";
					backupFiles(bss.str().c_str());
				}

				delete[] afterSecError;
				delete[] testAfterSecError;

				if (trainSamples == totalSamples && ((numRunsOnFullTrainset != 0 && numRuns - numRunSetStart >= numRunsOnFullTrainset) || (lastErrors.back() <= fullTrainsetErrorGoal))) {
					std::cout << "Run completed after " << numRuns - numRunSetStart << " rounds on full trainset" << std::endl;
					break;
				}
				if (!belowStallThresh && trainSamples == totalSamples && ((stallRestartThreshold != 0 && lastErrors.back() < stallRestartThreshold) || (minTestErrorStallRestartThreshold != 0 && minTestError < minTestErrorStallRestartThreshold))) {
					belowStallThresh = true;
				}
				if (!belowStallThresh && trainSamples == totalSamples && numRunsToStallRestart > 0 && numRuns - numRunSetStart >= numRunsToStallRestart) {
					std::cout << "Passed stall threshold after " << numRuns - numRunSetStart << " rounds, restarting with new weights." << std::endl;
					moveSaveToStallBackup();
					if (numTrainCrossValSets != 0 && cv > 0)
						cv--;
					else {
						bagNum--;
						if (numTrainCrossValSets != 0)
							cv = numTrainCrossValSets - 1;
					}
					skipDataSetLoad = true;
					break;
				}

				if (primaryAfterError < trainIncreaseThreshold && trainSamples < totalSamples) {
					saveSetHistory(numSamples, numRuns - numRunSetStart, stepMult);
					numRunSetStart = numRuns;
					trainSamples = min(trainSamplesIncreaseFactor * trainSamples, totalSamples);
					if (randomizeSubsetOnThreshold) {
						randomizeTrainSet();
						numSamples = trainSamples;
						std::cout << "Starting new run on " << numSamples << " samples" << std::endl;
					}
					else
						numSamples = readData(1, trainSamples);
					stepMult = max(stepMultDecFactor*stepMult, minimumStepMult);
					prevPrimaryAfterError = -1.0f;
				}

				saveSimVariables();
			}
			if (numTrainCrossValSets == 0)
				break;
		}
	}

	system("pause");
}

void saveResults(size_t numRuns, float afterError, float* afterSecError, float testAfterError, float* testAfterSecError, float testSampleAfterError) {
	std::stringstream resname;
	resname << savestring << savename << "result";
	std::ofstream resfile(resname.str().c_str(), std::ios_base::app);
	resfile << numRuns << " " << afterError;
	for (size_t i = 0; i < numSecErrors; i++) {
		resfile << " " << afterSecError[i];
	}
	resfile << " " << testAfterError;
	for (size_t i = 0; i < numSecErrors; i++) {
		resfile << " " << testAfterSecError[i];
	}

	if (testUseSampleFile)
		resfile << " " << testSampleAfterError;

	resfile << std::endl;

	if (trainOrderedByBin) {
		std::stringstream distname;
		distname << savestring << savename << "dist";
		std::ofstream distfile(distname.str().c_str(), std::ios_base::app);
		distfile << numRuns << " ";
		for (size_t i = 0; i < numBins; i++) {
			distfile << testBinFreqs[i] << " ";
		}
		distfile << std::endl;
	}
}

bool loadLastResults(float* afterError, float* afterSecError, float* testAfterError, float* testAfterSecError, float* testSampleAfterError) {
	std::stringstream resname;
	resname << savestring << savename << "result";
	std::ifstream resfile(resname.str().c_str());

	bool resultsExist = false;
	std::string line;
	while (getline(resfile, line)) {
		resultsExist = true;
		std::stringstream lss(line);
		std::string dum;
		lss >> dum;	//numRuns
		lss >> afterError[0];
		for (size_t i = 0; i < numSecErrors; i++) {
			lss >> afterSecError[i];
		}
		lss >> testAfterError[0];
		for (size_t i = 0; i < numSecErrors; i++) {
			lss >> testAfterSecError[i];
		}
		if (testUseSampleFile)
			lss >> testSampleAfterError[0];
	}
	return resultsExist;
}

void saveSetHistory(size_t nSamples, size_t nRuns, float stepFacMult) {
	std::stringstream hisname;
	hisname << savestring << savename << "history";
	std::ofstream hisfile(hisname.str().c_str(), std::ios_base::app);
	hisfile << nSamples << " " << stepFacMult << " " << nRuns << std::endl;
}

void loadSimVariables(bool userParams) {
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
		if (var == "stepAdjustment")
			lss >> stepAdjustment;
		if (var == "numRunSetStart")
			lss >> numRunSetStart;
		if (var == "minTestError")
			lss >> minTestError;
		if (!userParams && var == "testBinFreqs") {
			testBinFreqs.resize(numBins);
			for (size_t i = 0; i < numBins; i++)
				lss >> testBinFreqs[i];
		}
	}
}

void saveSimVariables() {
	std::stringstream pss;
	pss << savestring << savename << "pars";
	std::ofstream outfile(pss.str().c_str());

	outfile << "numRuns " << numRuns << std::endl;
	outfile << "trainSamples " << trainSamples << std::endl;
	outfile << "stepMult " << stepMult << std::endl;
	outfile << "stepAdjustment " << stepAdjustment << std::endl;
	outfile << "numRunSetStart " << numRunSetStart << std::endl;
	if (backupMinTestError)
		outfile << "minTestError " << minTestError << std::endl;
	if (trainOrderedByBin) {
		outfile << "testBinFreqs ";
		for (size_t i = 0; i < numBins; i++)
			outfile << testBinFreqs[i] << " ";
		outfile << std::endl;
	}
		
}

size_t readData(size_t begin, size_t numIOs, bool readTestSet) {
	if (numIOs > 0)
		std::cout << "Reading " << numIOs << " samples from trainset ";
	else
		std::cout << "Reading all samples from trainset ";

	auto readstart = std::chrono::high_resolution_clock::now();
#ifndef RAND_EXPLICIT
	std::cout << trainstring << ": ";
	totalSamples = readTrainSet(trainstring, begin, numIOs);
#else
	std::cout << randtrainstring << ": ";
	totalSamples = readExplicitTrainSet(randtrainstring, begin, numIOs);
#endif
	auto readelapsed = std::chrono::high_resolution_clock::now() - readstart;
	long long readtime = std::chrono::duration_cast<std::chrono::microseconds>(readelapsed).count();
	std::cout << readtime / 1000000 << " s" << std::endl;
	size_t numSamples;
	if (numIOs > 0)
		numSamples = min(numIOs, totalSamples);
	else
		numSamples = totalSamples;
	std::cout << numSamples << "/" << totalSamples << " samples loaded" << std::endl;

	if (readTestSet) {
		std::cout << "Reading all samples from testset " << testfile << ": ";

		auto readstart = std::chrono::high_resolution_clock::now();
		size_t testSamples;
		if (!testUseSampleFile) {
			if (!testExplicitFile)
				testSamples = readTrainSet(testfile, 1, 0, false, true);
			else
				testSamples = readExplicitTrainSet(testfile, 1, 0, true);
		}
		else {
			size_t numDiscards[2];
			sampleReadTrainSet(testfile, discardSamples, numDiscards, false, true);

			std::cout << numDiscards[0] << "/" << numDiscards[1] << " samples discarded" << std::endl;

			testSamples = numDiscards[1] - numDiscards[0];
		}

		auto readelapsed = std::chrono::high_resolution_clock::now() - readstart;
		long long readtime = std::chrono::duration_cast<std::chrono::microseconds>(readelapsed).count();
		std::cout << readtime / 1000000 << " s" << std::endl;
		std::cout << testSamples << " samples loaded" << std::endl;
	}
	return numSamples;
}

void loadLocalParameters() {
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
		else if (var == "annealingStartError")
			lss >> annealingStartError;
		else if (var == "initialTrainSamples")
			lss >> initialTrainSamples;
		else if (var == "trainSamplesIncreaseFactor")
			lss >> trainSamplesIncreaseFactor;
		else if (var == "trainIncreaseThreshold")
			lss >> trainIncreaseThreshold;
		else if (var == "backupInterval")
			lss >> backupInterval;
		else if (var == "backupSampleNumStart")
			lss >> backupSampleNumStart;
		else if (var == "randomizeSubsetOnThreshold")
			lss >> randomizeSubsetOnThreshold;
		else if (var == "randomizeTrainSetEveryRun")
			lss >> randomizeTrainSetEveryRun;
		else if (var == "pairedTraining")
			lss >> pairedTraining;
		else if (var == "redoErrorThreshold")
			lss >> redoErrorThreshold;
		else if (var == "redoStepAdjustment")
			lss >> redoStepAdjustment;
		else if (var == "successStepAdjustment")
			lss >> successStepAdjustment;
		else if (var == "stepAdjustmentNumStart")
			lss >> stepAdjustmentNumStart;
		else if (var == "printTestError")
			lss >> printTestError;
		else if (var == "minTrainSizeToPrintTestError")
			lss >> minTrainSizeToPrintTestError;
		else if (var == "testExplicitFile")
			lss >> testExplicitFile;
		else if (var == "testfile")
			lss >> testfile;
		else if (var == "discardSamples")
			lss >> discardSamples;
		else if (var == "testUseSampleFile")
			lss >> testUseSampleFile;
		else if (var == "nIter")
			lss >> nIter;
		else if (var == "backupMinTestError")
			lss >> backupMinTestError;
		else if (var == "trainStockBAGSet")
			lss >> trainStockBAGSet;
		else if (var == "trainBAGSubnetNum")
			lss >> trainBAGSubnetNum;
		else if (var == "numRunsOnFullTrainset")
			lss >> numRunsOnFullTrainset;
		else if (var == "fullTrainsetErrorGoal")
			lss >> fullTrainsetErrorGoal;
		else if (var == "primaryErrorType")
			lss >> primaryErrorType;
		else if (var == "testMinErrorType")
			lss >> testMinErrorType;
		else if (var == "numRunsToStallRestart")
			lss >> numRunsToStallRestart;
		else if (var == "stallRestartThreshold")
			lss >> stallRestartThreshold;
		else if (var == "addBagTradeSuffix")
			lss >> addBagTradeSuffix;
		else if (var == "numTrainCrossValSets")
			lss >> numTrainCrossValSets;
		else if (var == "minTestErrorStallRestartThreshold")
			lss >> minTestErrorStallRestartThreshold;
		else if (var == "backupMinTestErrorSample")
			lss >> backupMinTestErrorSample;
		else if (var == "resultSaveSampleNumStart")
			lss >> resultSaveSampleNumStart;
		else if (var == "declaredTotalSamples")
			lss >> declaredTotalSamples;
		else if (var == "binOutAverageTrainReg")
			lss >> binOutAverageTrainReg;
		else if (var == "appendBAGSubnetNum")
			lss >> appendBAGSubnetNum;
		else if (var == "multTrainSetsPerCV")
			lss >> multTrainSetsPerCV;
		else if (var == "fullTrainSetStepAdjustment")
			lss >> fullTrainSetStepAdjustment;
	}
}

void backupFiles(std::string backname) {
	std::stringstream bss;
	bss << savestring << "backup/" << backname;

	std::stringstream oss;
	oss << savestring << savename;

	std::stringstream pss;
	pss << oss.str();

	std::stringstream nss;
	nss << bss.str();

	CopyFile(pss.str().c_str(), nss.str().c_str(), false);

	if(pairedTraining) {
		pss.clear();
		pss.str("");
		nss.clear();
		nss.str("");

		pss << oss.str() << "conv";
		nss << bss.str() << "conv";

		CopyFile(pss.str().c_str(), nss.str().c_str(), false);

		pss.clear();
		pss.str("");
		nss.clear();
		nss.str("");

		pss << oss.str() << "fixed";
		nss << bss.str() << "fixed";

		CopyFile(pss.str().c_str(), nss.str().c_str(), false);
	}

	pss.clear();
	pss.str("");
	nss.clear();
	nss.str("");

	pss << oss.str() << "result";
	nss << bss.str() << "result";

	CopyFile(pss.str().c_str(), nss.str().c_str(), false);

	pss.clear();
	pss.str("");
	nss.clear();
	nss.str("");

	pss << oss.str() << "pars";
	nss << bss.str() << "pars";

	CopyFile(pss.str().c_str(), nss.str().c_str(), false);

	pss.clear();
	pss.str("");
	nss.clear();
	nss.str("");

	pss << oss.str() << "history";
	nss << bss.str() << "history";

	CopyFile(pss.str().c_str(), nss.str().c_str(), false);
}

void moveSaveToStallBackup() {
	std::stringstream basestallss;
	basestallss << savestring << "lastStall";
	std::stringstream basess;
	basess << savestring << savename;

	std::stringstream stall;
	stall << basestallss.str();
	std::stringstream ss;
	ss << basess.str();
	DeleteFile(stall.str().c_str());
	MoveFile(ss.str().c_str(), stall.str().c_str());

	stall.str("");
	stall.clear();
	ss.str("");
	ss.clear();

	stall << basestallss.str() << "pars";
	ss << basess.str() << "pars";
	DeleteFile(stall.str().c_str());
	MoveFile(ss.str().c_str(), stall.str().c_str());

	stall.str("");
	stall.clear();
	ss.str("");
	ss.clear();

	stall << basestallss.str() << "result";
	ss << basess.str() << "result";
	DeleteFile(stall.str().c_str());
	MoveFile(ss.str().c_str(), stall.str().c_str());

	stall.str("");
	stall.clear();
	ss.str("");
	ss.clear();

	stall << basestallss.str() << "history";
	ss << basess.str() << "history";
	DeleteFile(stall.str().c_str());
	MoveFile(ss.str().c_str(), stall.str().c_str());
}