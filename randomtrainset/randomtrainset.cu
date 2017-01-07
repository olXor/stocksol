#include "stockrun.cuh"

#ifdef LOCAL
#define datastring "rawdata/"
#define savestring "saveweights/"
#else
#define datastring "../stockproj/rawdata/"
#define savestring "../stockproj/saveweights/"
#endif

bool testUseSampleFile = false;
bool discardSamples = false;

bool ranTrainUseSampleFile = false;

bool createRandomizedExplicitTrainSetFile = true;

void loadLocalParameters(std::string parName);

bool ranTrainCrossVal = false;
size_t numRanTrainCrossSets = 2;
size_t ranTrainCrossGapSize = 0;

void randomizeDataSet(std::vector<IOPair>* trainset, size_t maxIndex = 0);
void saveExplicitDataSet(std::vector<IOPair>* trainset, std::string learnsetname);
void saveCrossTestSegment(std::string filename, size_t cvnum, size_t testBegin, size_t testEnd);

bool saveRanTrainCrossTestSegments = true;

int main() {
	srand((size_t)time(NULL));
	setStrings(datastring, savestring);
#ifdef LOCAL
	loadParameters("pars.cfg");
	loadLocalParameters("pars.cfg");
#else
	loadParameters("../stockproj/pars.cfg");
	loadLocalParameters("../stockproj/pars.cfg");
#endif

	size_t numSamples;
	std::cout << "Reading trainset " << trainstring << ": ";
	if (ranTrainUseSampleFile) {
		auto readstart = std::chrono::high_resolution_clock::now();
		size_t numDiscards[2];

		sampleReadTrainSet(trainstring, discardSamples, numDiscards, true);

		auto readelapsed = std::chrono::high_resolution_clock::now() - readstart;
		long long readtime = std::chrono::duration_cast<std::chrono::microseconds>(readelapsed).count();
		std::cout << readtime / 1000000 << " s" << std::endl;

		std::cout << numDiscards[0] << "/" << numDiscards[1] << " samples discarded" << std::endl;

		numSamples = numDiscards[1] - numDiscards[0];
	}
	else {
		auto readstart = std::chrono::high_resolution_clock::now();

		numSamples = readTrainSet(trainstring, 1, 0, true);

		auto readelapsed = std::chrono::high_resolution_clock::now() - readstart;
		long long readtime = std::chrono::duration_cast<std::chrono::microseconds>(readelapsed).count();
		std::cout << readtime / 1000000 << " s" << std::endl;
	}
	std::cout << numSamples << " samples read" << std::endl;

	if (ranTrainCrossVal) {
		std::cout << "Creating " << numRanTrainCrossSets << " cross-validation sets" << std::endl;
		for (size_t i = 0; i < numRanTrainCrossSets; i++) {
			std::vector<IOPair>* fullset = getTrainSet();
			std::vector<IOPair> crosstrain;
			std::vector<IOPair> crosstest;
			crosstrain.clear();
			crosstest.clear();
			size_t trainGapBegin = (size_t)(fullset->size()*(1.0f*i / numRanTrainCrossSets));
			size_t trainGapEnd = (size_t)(fullset->size()*(1.0f*(i + 1) / numRanTrainCrossSets));
			size_t testBegin = (size_t)(fullset->size()*(1.0f*i / numRanTrainCrossSets));
			size_t testEnd = (size_t)(fullset->size()*(1.0f*(i + 1) / numRanTrainCrossSets));
			if (i != 0) {
				trainGapBegin -= ranTrainCrossGapSize / 2;
				testBegin += ranTrainCrossGapSize / 2;
			}
			if (i != numRanTrainCrossSets - 1) {
				trainGapEnd += ranTrainCrossGapSize / 2;
				testEnd -= ranTrainCrossGapSize / 2;
			}
			if (trainGapBegin > fullset->size() || testBegin > fullset->size() || trainGapEnd > fullset->size() || testEnd > fullset->size()) {
				std::cout << "Cross validation gap size invalid!" << std::endl;
				system("pause");
				return 0;
			}

			if (saveRanTrainCrossTestSegments && !ranTrainUseSampleFile) {
				saveCrossTestSegment(trainstring, i+1, testBegin, testEnd);
			}
			std::cout << "Set #" << i + 1 << ": Train Gap from " << trainGapBegin << "-" << trainGapEnd << " Test Set from " << testBegin << "-" << testEnd << std::endl;

			for (size_t j = 0; j < fullset->size(); j++) {
				if (j < trainGapBegin || j > trainGapEnd)
					crosstrain.push_back((*fullset)[j]);
				else if (j >= testBegin && j < testEnd)
					crosstest.push_back((*fullset)[j]);
			}

			if (createRandomizedExplicitTrainSetFile) {
				randomizeDataSet(&crosstrain);
				randomizeDataSet(&crosstest);
			}

			std::stringstream trainss;
			trainss << randtrainstring << "CrossTrain" << i + 1;
			std::stringstream testss;
			testss << randtrainstring << "CrossTest" << i + 1;
			saveExplicitDataSet(&crosstrain, trainss.str());
			saveExplicitDataSet(&crosstest, testss.str());
		}
	}
	else {
		if (createRandomizedExplicitTrainSetFile)
			randomizeTrainSet();

		saveExplicitTrainSet(randtrainstring);
	}

#ifdef LOCAL
	system("pause");
#endif
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
		else if (var == "discardSamples")
			lss >> discardSamples;
		else if (var == "createRandomizedExplicitTrainSetFile")
			lss >> createRandomizedExplicitTrainSetFile;
		else if (var == "ranTrainUseSampleFile")
			lss >> ranTrainUseSampleFile;
		else if (var == "ranTrainCrossVal")
			lss >> ranTrainCrossVal;
		else if (var == "numRanTrainCrossSets")
			lss >> numRanTrainCrossSets;
		else if (var == "ranTrainCrossGapSize")
			lss >> ranTrainCrossGapSize;
		else if (var == "saveRanTrainCrossTestSegments")
			lss >> saveRanTrainCrossTestSegments;
	}
}

void randomizeDataSet(std::vector<IOPair>* trainset, size_t maxIndex) {
	if (maxIndex == 0 || maxIndex > (*trainset).size())
		maxIndex = (*trainset).size();
	for (size_t i = 0; i < maxIndex; i++) {
		size_t j = rand() % maxIndex;
		IOPair tmpio = (*trainset)[i];
		(*trainset)[i] = (*trainset)[j];
		(*trainset)[j] = tmpio;
	}
}

void saveExplicitDataSet(std::vector<IOPair>* trainset, std::string learnsetname) {
	std::stringstream learnsetss;
	learnsetss << datastring << learnsetname;
	std::ofstream outfile(learnsetss.str().c_str());

	for (size_t i = 0; i < (*trainset).size(); i++) {
		outfile << (*trainset)[i].correctoutput*OUTPUT_DIVISOR << " | ";
		for (size_t j = 0; j < (*trainset)[i].inputs.size(); j++) {
			outfile << (*trainset)[i].inputs[j] << " ";
		}
		outfile << std::endl;
	}
}

void saveCrossTestSegment(std::string filename, size_t cvnum, size_t testBegin, size_t testEnd) {
	testEnd += NUM_INPUTS-1;	//to change from intervals to data points
	std::stringstream fss;
	fss << datastring << filename;
	std::ifstream infile(fss.str());
	std::stringstream outss;
	outss << datastring << filename << "CrossTest" << cvnum;
	std::ofstream outfile(outss.str());

	std::string line;
	size_t linenum = 0;
	while (getline(infile, line)) {
		linenum++;
		if (linenum > testBegin && linenum < testEnd)
			outfile << line << std::endl;
	}
}
