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

void saveResults(size_t numRuns, float afterError, float* afterSecError, float testAfterError, float* testAfterSecError);
void saveSetHistory(size_t nSamples, size_t nRuns, float stepFacMult);
void loadLocalParameters();
void loadSimVariables();
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

bool trainStockBAGSet = false;
size_t trainBAGSubnetNum = 1;
bool addBagTradeSuffix = false;

size_t numRunsOnFullTrainset = 0;
float fullTrainsetErrorGoal = 0.0f;
size_t numRunsToStallRestart = 0;
float stallRestartThreshold = 0.0f;

void moveSaveToStallBackup();

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

	size_t numBagSubnets = 1;
	if (trainStockBAGSet) {
		if (addBagTradeSuffix)
			numBagSubnets = 2 * trainBAGSubnetNum;
		else
			numBagSubnets = trainBAGSubnetNum;
	}
	//break out of the loop after one iteration if trainStockBAGSet = false
	for (size_t bagNum = 0; bagNum < numBagSubnets; bagNum++) {
		stepAdjustment = 1.0f;
		loadParameters("pars.cfg");
		loadLocalParameters();
		if (trainStockBAGSet) {
			if (addBagTradeSuffix) {
				if (bagNum % 2 == 0) {
					std::stringstream savess;
					savess << savename << "long" << bagNum / 2 + 1;
					savename = savess.str();
					std::stringstream rantrainss;
					rantrainss << randtrainstring << "long" << bagNum / 2 + 1;
					randtrainstring = rantrainss.str();
					std::stringstream testss;
					testss << testfile << "long" << bagNum / 2 + 1;
					testfile = testss.str();
				}
				else {
					std::stringstream savess;
					savess << savename << "short" << bagNum / 2 + 1;
					savename = savess.str();
					std::stringstream rantrainss;
					rantrainss << randtrainstring << "short" << bagNum / 2 + 1;
					randtrainstring = rantrainss.str();
					std::stringstream testss;
					testss << testfile << "short" << bagNum / 2 + 1;
					testfile = testss.str();
				}
			}
			else {
				std::stringstream savess;
				savess << savename  << bagNum + 1;
				savename = savess.str();
				std::stringstream rantrainss;
				rantrainss << randtrainstring << bagNum + 1;
				randtrainstring = rantrainss.str();
				std::stringstream testss;
				testss << testfile << bagNum + 1;
				testfile = testss.str();
			}
			std::cout << std::endl << "Beginning run on " << savename << std::endl;
		}
		numRuns = 0;
		trainSamples = initialTrainSamples;
		stepMult = initialStepMult;
		numRunSetStart = 0;
		lastErrors.clear();
		minTestError = 9999999.0f;

		loadSimVariables();

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

		std::cout << "Calculating initial error: ";
		float initError;
		float* initSecError = new float[numSecErrors];
		for (size_t i = 0; i < numSecErrors; i++)
			initSecError[i] = 0.0f;
		float testInitError;
		float* testInitSecError = new float[numSecErrors];
		for (size_t i = 0; i < numSecErrors; i++)
			testInitSecError[i] = 0.0f;

		disableDropout();
		auto initstart = std::chrono::high_resolution_clock::now();
		if (pairedTraining)
			initError = runPairedSim(pairedLayers, false, 0, trainSamples);
		else {
			initError = runSim(layers, false, 0, trainSamples, false, initSecError);
		}
		if (printTestError && !pairedTraining) {
			testInitError = runSim(layers, false, 0, 0, false, testInitSecError, true);
		}

		auto initelapsed = std::chrono::high_resolution_clock::now() - initstart;
		long long inittime = std::chrono::duration_cast<std::chrono::microseconds>(initelapsed).count();

		std::cout << inittime / 1000000 << " s, Error: " << initError;
		for (size_t i = 0; i < numSecErrors; i++) {
			if (initSecError[i] != 0.0f)
				std::cout << " SE" << i + 1 << ": " << initSecError[i];
		}
		if (printTestError && trainSamples >= minTrainSizeToPrintTestError && !pairedTraining) {
			std::cout << " | Test Error: " << testInitError;
			for (size_t i = 0; i < numSecErrors; i++) {
				if (testInitSecError[i] != 0.0f)
					std::cout << " SE" << i + 1 << ": " << testInitSecError[i];
			}
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

		while (true) {
			if (trainSamples == totalSamples && ((numRunsOnFullTrainset != 0 && numRuns - numRunSetStart >= numRunsOnFullTrainset) || (lastErrors.back() <= fullTrainsetErrorGoal))) {
				std::cout << "Run completed after " << numRuns - numRunSetStart << " rounds on full trainset" << std::endl;
				break;
			}
			if (trainSamples == totalSamples && numRunsToStallRestart > 0 && numRuns - numRunSetStart >= numRunsToStallRestart && (stallRestartThreshold == 0 || lastErrors.back() > stallRestartThreshold)) {
				std::cout << "Passed stall threshold after " << numRuns - numRunSetStart << " rounds, restarting with new weights." << std::endl;
				moveSaveToStallBackup();
				bagNum--;
				break;
			}
			if (pairedTraining)
				std::cout << nIter << "+1 runs on " << numSamples << " sample pairs";
			else
				std::cout << nIter << "+1 runs on " << numSamples << " samples";
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
					runSim(layers, true, stepMultiplier(numRuns), trainSamples);
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
			if (printTestError && trainSamples >= minTrainSizeToPrintTestError && !pairedTraining) {
				for (size_t i = 0; i < numSecErrors; i++)
					testAfterSecError[i] = 0.0f;
				testAfterError = runSim(layers, false, 0, 0, false, testAfterSecError, true);
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
			}
			std::cout << std::endl;

			loadSimVariables();	//to allow the user to change variables mid-run

			float primaryAfterError;
			if (primaryErrorType == 0)
				primaryAfterError = afterError;
			else
				primaryAfterError = afterSecError[primaryErrorType - 1];

			if (trainSamples >= stepAdjustmentNumStart) {
				if (redoErrorThreshold > 0.0f && prevPrimaryAfterError > 0.0f && primaryAfterError - prevPrimaryAfterError > redoErrorThreshold) {
					std::cout << "Error increase was above threshold; redoing last run with lower stepfactor" << std::endl;

					if (pairedTraining)
						loadPairedWeights(pairedLayers, savename);
					else
						loadWeights(layers, savename);
					stepAdjustment *= redoStepAdjustment;
					continue;
				}
				stepAdjustment *= successStepAdjustment;
			}
			prevPrimaryAfterError = primaryAfterError;
			updateLastErrors(primaryAfterError);
			if (pairedTraining)
				savePairedWeights(pairedLayers, savename);
			else
				saveWeights(layers, savename);
			numRuns += nIter;

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

			saveResults(numRuns, afterError, afterSecError, testAfterError, testAfterSecError);

			if (trainSamples >= backupSampleNumStart && backupInterval > 0 && (numRuns - numRunSetStart) % backupInterval == 0) {
				std::stringstream bss;
				bss << savename << numSamples << "-" << numRuns - numRunSetStart;
				backupFiles(bss.str().c_str());
			}

			float testMinAfterError;
			if (testMinErrorType == 0)
				testMinAfterError = testAfterError;
			else
				testMinAfterError = testAfterSecError[testMinErrorType - 1];
			if (backupMinTestError && testMinAfterError < minTestError) {
				minTestError = testMinAfterError;
				std::stringstream bss;
				bss << savename << "Min";
				backupFiles(bss.str().c_str());
			}

			delete[] afterSecError;
			delete[] testAfterSecError;

			saveSimVariables();
		}
		if (!trainStockBAGSet && !(trainSamples == totalSamples && numRunsToStallRestart > 0 && numRuns - numRunSetStart >= numRunsToStallRestart))
			break;
	}
}

void saveResults(size_t numRuns, float afterError, float* afterSecError, float testAfterError, float* testAfterSecError) {
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
	resfile << std::endl;
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
		if (var == "minTestError")
			lss >> minTestError;
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
	if (backupMinTestError)
		outfile << "minTestError " << minTestError << std::endl;
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