#include <stockrun.cuh>
#include "Shlwapi.h"

#ifdef LOCAL
#define datastring "rawdata/"
#define savestring "saveweights/"
#else
#define datastring "../stockproj/rawdata/"
#define savestring "../stockproj/saveweights/"
#endif

bool testExplicitFile = false;
size_t testBegin = 1;
size_t testNumIOs = 0;

bool testUseSampleFile = false;
bool discardSamples = false;
std::string testOutputFile = "testoutput";
std::string testfile = "trainset";

bool testPrintSampleAll = false;

bool pairedTraining = false;
size_t testPairsAveraged = 0;

bool testBackups = false;
size_t backupStartRound = 0;
size_t backupInterval = 100;
size_t testBackupSampleSize = 2;
std::string testBackupHistoryLog = "backupHistoryLog";

bool testMinBackup = false;

bool testPrint = true;

size_t readData(size_t begin, size_t numIOs);
void loadLocalParameters(std::string parName);

int main() {
	srand((size_t)time(NULL));
	setStrings(datastring, savestring);
#ifdef LOCAL
	loadLocalParameters("pars.cfg");
	loadParameters("pars.cfg");
#else
	loadLocalParameters("../stockproj/pars.cfg");
	loadParameters("../stockproj/pars.cfg");
#endif

	readData(testBegin, testNumIOs);

	std::string basename;
	size_t backupNum = backupStartRound;
	std::ofstream backupHistoryLog;
	if (testBackups) {
		basename = savename;
		backupHistoryLog.open(testBackupHistoryLog);
	}

	float error;
	do {
		if (testBackups) {
			std::stringstream savess;
			savess << "backup/" << basename << testBackupSampleSize << "-" << backupNum;
			savename = savess.str();
		}
		else if (testMinBackup) {
			std::stringstream savess;
			savess << "backup/" << savename << "Min";
			savename = savess.str();
		}
		LayerCollection layers;
		PairedConvCollection pairedLayers;
		if (pairedTraining) {
			pairedLayers = createAndInitializePairedConvCollection(NUM_INPUTS);
			if (!loadPairedWeights(pairedLayers, savename)) {
#ifdef LOCAL
				system("pause");
#endif
				return 0;
			}
		}
		else {
			layers = createLayerCollection(0, getLCType());
			initializeLayers(&layers);
			if (!loadWeights(layers, savename)) {
#ifdef LOCAL
				system("pause");
#endif
				return 0;
			}
		}

		disableDropout();		//don't drop neurons while testing.
		std::cout << "Starting test: " << std::endl;
		auto teststart = std::chrono::high_resolution_clock::now();
		if (pairedTraining) {
			if (testUseSampleFile) {
				std::cout << "Can't test with sample file when using paired training" << std::endl;
				throw std::runtime_error("Can't test with sample file when using paired training");
			}
			else
				error = runPairedSim(pairedLayers, false, 0, 0, testPrint, testPairsAveraged);
		}
		else {
			if (testUseSampleFile) {
				std::ofstream outfile;
				if (testOutputFile != "")
					outfile.open(testOutputFile);
				if (INTERVALS_PER_DATASET == 0)
					error = sampleTestSim(layers, &outfile, testPrintSampleAll);
				else {
					error = 0.0f;
					size_t intervalNum = testBegin;
					size_t numSamples = 0;
					size_t numLoaded = 0;
					do {
						size_t trainSize = getTrainSet()->size();
						error += trainSize*sampleTestSim(layers, &outfile, testPrintSampleAll);
						intervalNum += INTERVALS_PER_DATASET;
						numSamples += trainSize;
						numLoaded = readData(intervalNum, 0);
					} while (numLoaded != 0);
					if (numSamples != 0)
						error /= numSamples;
				}
			}
			else {
				std::ofstream outfile;
				if (testOutputFile != "")
					outfile.open(testOutputFile);
				error = runSim(layers, false, 0, 0, testPrint, NULL, false, NULL, &outfile);
			}
		}

		auto testelapsed = std::chrono::high_resolution_clock::now() - teststart;
		long long testtime = std::chrono::duration_cast<std::chrono::microseconds>(testelapsed).count();
		std::cout << "Test took " << testtime / 1000000 << " s" << std::endl;

		if (testBackups) {
			std::cout << "Round " << backupNum << " error: " << error << std::endl;
			backupHistoryLog << backupNum << " " << error << std::endl;
			backupNum += backupInterval;
		}
		else
			std::cout << "Average Error " << error << std::endl;
	} while (testBackups);

#ifdef LOCAL
	system("pause");
#endif
}

size_t readData(size_t begin, size_t numIOs) {
	size_t numSamples;
	if (!testUseSampleFile) {
		if (numIOs > 0)
			std::cout << "Reading " << numIOs << " samples from trainset " << testfile << ": ";
		else
			std::cout << "Reading all samples from trainset " << testfile << ": ";

		auto readstart = std::chrono::high_resolution_clock::now();
		size_t totalSamples;
		if (!testExplicitFile)
			totalSamples = readTrainSet(testfile, begin, numIOs);
		else
			totalSamples = readExplicitTrainSet(testfile, begin, numIOs);

		auto readelapsed = std::chrono::high_resolution_clock::now() - readstart;
		long long readtime = std::chrono::duration_cast<std::chrono::microseconds>(readelapsed).count();
		std::cout << readtime / 1000000 << " s" << std::endl;
		size_t numSamples;
		if (numIOs > 0)
			numSamples = min(numIOs, totalSamples);
		else
			numSamples = totalSamples;
		std::cout << numSamples << "/" << totalSamples << " samples loaded" << std::endl;
		return numSamples;
	}
	else {
		if (INTERVALS_PER_DATASET > 0)
			std::cout << "Reading " << INTERVALS_PER_DATASET << " intervals from trainset " << testfile << " starting at " << begin << ": ";
		else
			std::cout << "Reading trainset " << testfile << ": ";
		auto readstart = std::chrono::high_resolution_clock::now();
		size_t numDiscards[2];

		sampleReadTrainSet(testfile, discardSamples, numDiscards, false, false, begin, true);

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
		else if (var == "testExplicitFile")
			lss >> testExplicitFile;
		else if (var == "testfile")
			lss >> testfile;
		else if (var == "testPrintSampleAll")
			lss >> testPrintSampleAll;
		else if (var == "pairedTraining")
			lss >> pairedTraining;
		else if (var == "testPairsAveraged")
			lss >> testPairsAveraged;
		else if (var == "testBackups")
			lss >> testBackups;
		else if (var == "backupStartRound")
			lss >> backupStartRound;
		else if (var == "backupInterval")
			lss >> backupInterval;
		else if (var == "testPrint")
			lss >> testPrint;
		else if (var == "testBackupSampleSize")
			lss >> testBackupSampleSize;
		else if (var == "testBackupHistoryLog")
			lss >> testBackupHistoryLog;
		else if (var == "testMinBackup")
			lss >> testMinBackup;
	}
}
