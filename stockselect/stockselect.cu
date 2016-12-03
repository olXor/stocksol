#include "stockrun.cuh"
#include <random>

#ifdef LOCAL
#define datastring "rawdata/"
#define savestring "saveweights/"
#else
#define datastring "../stockproj/rawdata/"
#define savestring "../stockproj/saveweights/"
#endif

std::string selecttestfile = "testfile";

struct SelectionCriteria {
	size_t minSubnetSelect;
	size_t oppositeMinSubnetSelect;
	std::vector<float> testSelectBinMins;
	std::vector<float> testSelectBinMaxes;
	std::vector<float> oppositeSelectBinMins;
	std::vector<float> oppositeSelectBinMaxes;
};

std::vector < std::vector<std::vector<float>>> subnetResults;	//[dataset.size(), 2*numSubnets, numBins]

bool testExplicitFile = false;
size_t testBegin = 1;
size_t testNumIOs = 0;

bool testUseSampleFile = false;
bool discardSamples = false;

void loadLocalParameters(std::string parName);

size_t numSubnets = 1;
size_t minSubnetSelect = 1;	//the number of subnets that need to return a positive result to select
size_t oppositeMinSubnetSelect = 1;

std::vector<LayerCollection> longsubnets;
std::vector<LayerCollection> shortsubnets;

bool longSelection = true;
bool shortSelection = false;
bool randomizeSelectOptimizationStartPoint = true;

size_t selectionEvaluationType = 0;

float selectPerturbSigma = 2.0f;

size_t readData(size_t begin, size_t numIOs);
float evaluateSelectionCriteria(SelectionCriteria crit, bool print);
SelectionCriteria getRandomSelectionCriteria();
SelectionCriteria perturbSelectionCriteria(SelectionCriteria crit);

void saveSelectionCriteria(SelectionCriteria crit);
bool loadSelectionCriteria(SelectionCriteria* crit);
void printSelectionCriteria(SelectionCriteria crit);
void generateSubnetResults();

int main() {
	srand((size_t)time(NULL));

	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
	if (!prop.canMapHostMemory)
		exit(0);
	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));

#ifdef LOCAL
	loadLocalParameters("pars.cfg");
	loadParameters("pars.cfg");
#else
	loadLocalParameters("../stockproj/pars.cfg");
	loadParameters("../stockproj/pars.cfg");
#endif

	setStrings(datastring, savestring);

	readData(testBegin, testNumIOs);

	longsubnets.resize(numSubnets);
	shortsubnets.resize(numSubnets);
	for (size_t i = 0; i < numSubnets; i++) {
		if (longSelection) {
			longsubnets[i] = createLayerCollection(0, getLCType());
			initializeLayers(&longsubnets[i]);

			std::stringstream wss;
			wss << savename << "long" << i + 1;
			if (!loadWeights(longsubnets[i], wss.str().c_str())) {
				std::cout << "couldn't find long weights file #" << i + 1 << std::endl;
#ifdef LOCAL
				system("pause");
#endif
				return 0;
			}
		}
		if (shortSelection) {
			shortsubnets[i] = createLayerCollection(0, getLCType());
			initializeLayers(&shortsubnets[i]);

			std::stringstream wss;
			wss << savename << "short" << i + 1;
			if (!loadWeights(shortsubnets[i], wss.str().c_str())) {
				std::cout << "couldn't find short weights file #" << i + 1 << std::endl;
#ifdef LOCAL
				system("pause");
#endif
				return 0;
			}
		}
	}

	SelectionCriteria currentCrit;
	if (!loadSelectionCriteria(&currentCrit)) {
		if (randomizeSelectOptimizationStartPoint)
			currentCrit = getRandomSelectionCriteria();
		else {
			currentCrit.minSubnetSelect = minSubnetSelect;
			currentCrit.oppositeMinSubnetSelect = oppositeMinSubnetSelect;
			currentCrit.testSelectBinMins = testSelectBinMins;
			currentCrit.testSelectBinMaxes = testSelectBinMaxes;
			currentCrit.oppositeSelectBinMins = oppositeSelectBinMins;
			currentCrit.oppositeSelectBinMaxes = oppositeSelectBinMaxes;
		}
	}

	std::cout << "Generating subnet results: ";
	auto genstart = std::chrono::high_resolution_clock::now();
	generateSubnetResults();
	auto genelapsed = std::chrono::high_resolution_clock::now() - genstart;
	long long gentime = std::chrono::duration_cast<std::chrono::microseconds>(genelapsed).count();
	std::cout << " (" << gentime / 1000000 << " s)" << std::endl;

	float currentBest = -999999.0f;
	SelectionCriteria testCrit = currentCrit;
	while (true) {
		float testEval = evaluateSelectionCriteria(testCrit, false);

		if (testEval > currentBest || currentBest == -999999.0f) {
			currentBest = testEval;
			currentCrit = testCrit;
			printSelectionCriteria(currentCrit);
			evaluateSelectionCriteria(currentCrit, true);
			saveSelectionCriteria(currentCrit);
		}
		testCrit = perturbSelectionCriteria(currentCrit);
	}
}

void generateSubnetResults() {
	std::vector<float*> d_inputs(2*numSubnets);
	std::vector<float*> h_output(2*numSubnets), d_output(2*numSubnets);
	for (size_t i = 0; i < 2*numSubnets; i++) {
		LayerCollection layers;
		size_t subnetPos = i % numSubnets;
		if (i < numSubnets)
			layers = longsubnets[subnetPos];
		else
			layers = shortsubnets[subnetPos];

		if (layers.numConvolutions > 0) {
			if (layers.convPars[0].numInputLocs != NUM_INPUTS || layers.convPars[0].numInputNeurons != 1)
				throw std::runtime_error("inputs to first layer don't match data set");
			d_inputs[i] = layers.convMat[0].inlayer;
		}
		else if (layers.numFixedNets > 0) {
			if (layers.fixedPars[0].numInputNeurons != NUM_INPUTS)
				throw std::runtime_error("inputs to first layer don't match data set");
			d_inputs[i] = layers.fixedMat[0].inlayer;
		}
		else
			throw std::runtime_error("tried to run on a network with no convolutions and no fixed networks");

		checkCudaErrors(cudaHostAlloc(&h_output[i], numBins*sizeof(float), cudaHostAllocMapped));
		checkCudaErrors(cudaHostGetDevicePointer(&d_output[i], h_output[i], 0));

		disableDropout();
		generateDropoutMask(&layers);
	}

	cudaStream_t mainStream = 0;

	std::vector<IOPair>* dataset = getTrainSet();
	subnetResults.resize(dataset->size());
	for (size_t i = 0; i < subnetResults.size(); i++) {
		subnetResults[i].resize(2 * numSubnets);
		for (size_t j = 0; j < subnetResults[i].size(); j++) {
			subnetResults[i][j].resize(numBins);
		}
	}

	for (size_t i = 0; i < dataset->size(); i++) {
		for (size_t j = 0; j < 2 * numSubnets; j++) {
			LayerCollection layers;
			size_t subnetPos = i % numSubnets;
			if (i < numSubnets)
				layers = longsubnets[subnetPos];
			else
				layers = shortsubnets[subnetPos];

			checkCudaErrors(cudaMemcpyAsync(d_inputs[j], &(*dataset)[i].inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice, mainStream));

			calculate(layers, mainStream);

			checkCudaErrors(cudaMemcpyAsync(layers.correctoutput, &(*dataset)[i].correctbins[0], numBins*sizeof(float), cudaMemcpyHostToDevice, mainStream));

			calculateOutputError << <1, numBins, 0, mainStream >> >(layers.d_fixedMat[layers.numFixedNets - 1], layers.stepfactor, layers.correctoutput, d_output[j]);
			checkCudaErrors(cudaPeekAtLastError());

			for (size_t k = 0; k < numBins; k++)
				subnetResults[i][j][k] = h_output[j][k];
		}
	}
}

float evaluateSelectionCriteria(SelectionCriteria crit, bool print) {
	std::vector<IOPair>* dataset = getTrainSet();
	float longProfit = 0.0f;
	float shortProfit = 0.0f;
	size_t numLongTrades = 0;
	size_t numShortTrades = 0;
	std::vector<size_t> longDist(numBins);
	std::vector<size_t> shortDist(numBins);
	for (size_t i = 0; i < subnetResults.size(); i++) {
		size_t numLongTestSelected = 0;
		size_t numLongOppositeSelected = 0;
		size_t numShortTestSelected = 0;
		size_t numShortOppositeSelected = 0;
		for (size_t j = 0; j < 2 * numSubnets; j++) {
			bool testSelected = true;
			bool oppositeSelected = true;
			for (size_t k = 0; k < numBins; k++) {
				if (subnetResults[i][j][k] < crit.testSelectBinMins[k] || subnetResults[i][j][k] > crit.testSelectBinMaxes[k])
					testSelected = false;
				if (subnetResults[i][j][k] < crit.oppositeSelectBinMins[k] || subnetResults[i][j][k] > crit.oppositeSelectBinMaxes[k])
					oppositeSelected = false;
			}

			if (j < numSubnets) { //LONG
				if (testSelected)
					numLongTestSelected++;
				if (oppositeSelected)
					numShortOppositeSelected++;
			}
			else {
				if (testSelected)
					numShortTestSelected++;
				if (oppositeSelected)
					numLongOppositeSelected++;
			}
		}
		if (numLongTestSelected >= crit.minSubnetSelect && numLongOppositeSelected >= crit.oppositeMinSubnetSelect) {
			longProfit += (*dataset)[i].correctoutput;
			numLongTrades++;
			size_t binPos = 0;
			for (size_t j = 0; j < numBins; j++) {
				if ((*dataset)[i].correctbins[j] == BIN_POSITIVE_OUTPUT) {
					binPos = j;
					break;
				}
			}
			longDist[binPos]++;
		}
		if (numShortTestSelected >= crit.minSubnetSelect && numShortOppositeSelected >= crit.oppositeMinSubnetSelect) {
			shortProfit += (*dataset)[i].secondaryoutput;
			numShortTrades++;
			size_t binPos = 0;
			for (size_t j = 0; j < numBins; j++) {
				if ((*dataset)[i].secondarybins[j] == BIN_POSITIVE_OUTPUT) {
					binPos = j;
					break;
				}
			}
			shortDist[binPos]++;
		}
	}

	if (print) {
		std::cout << "Profits: L: " << longProfit << " (/" << numLongTrades << "=" << longProfit / numLongTrades << ") S: " << shortProfit << " (/" << numShortTrades << "=" << shortProfit / numShortTrades << ")" << " Total: " << longProfit + shortProfit << " (/" << numLongTrades + numShortTrades << "=" << (longProfit + shortProfit) / (numLongTrades + numShortTrades) << ")" << std::endl;
		std::cout << "Long Trade Distribution: ";
		for (size_t i = 0; i < numBins; i++) {
			std::cout << longDist[i] << " ";
		}
		std::cout << std::endl;

		std::cout << "Short Trade Distribution: ";
		for (size_t i = 0; i < numBins; i++) {
			std::cout << shortDist[i] << " ";
		}
		std::cout << std::endl;
	}

	float evaluation = 0.0f;
	if (selectionEvaluationType == 0)
		evaluation = longProfit + shortProfit;
	else if (selectionEvaluationType == 1 && numLongTrades + numShortTrades != 0)
		evaluation = (longProfit + shortProfit)*fabs(longProfit + shortProfit) / (numLongTrades + numShortTrades);
	else if (selectionEvaluationType == 2 && numLongTrades + numShortTrades != 0)
		evaluation = (longProfit + shortProfit) / (numLongTrades + numShortTrades);
	else if (selectionEvaluationType == 3 && numLongTrades + numShortTrades != 0)
		evaluation = (float)(longProfit + shortProfit)*fabs(longProfit + shortProfit) / sqrt(numLongTrades + numShortTrades)/10.0f;
	else
		evaluation = 0.0f;

	if (print)
		std::cout << "Evaluation: " << evaluation << std::endl;

	return evaluation;
}

void loadLocalParameters(std::string parName) {
	std::ifstream infile(parName.c_str());
	std::string line;
	while (getline(infile, line)) {
		std::stringstream lss(line);
		std::string var;
		lss >> var;

		if (var == "numSubnets")
			lss >> numSubnets;
		else if (var == "minSubnetSelect")
			lss >> minSubnetSelect;
		else if (var == "oppositeMinSubnetSelect")
			lss >> oppositeMinSubnetSelect;
		else if (var == "longSelection")
			lss >> longSelection;
		else if (var == "shortSelection")
			lss >> shortSelection;
		else if (var == "testSelectBinMins") {
			while(!lss.eof()) {
				float binMin = -99999.0f;
				lss >> binMin;
				testSelectBinMins.push_back(binMin);
			}
		}
		else if (var == "testSelectBinMaxes") {
			while (!lss.eof()) {
				float binMax = 99999.0f;
				lss >> binMax;
				testSelectBinMaxes.push_back(binMax);
			}
		}
		else if (var == "oppositeSelectBinMins") {
			while(!lss.eof()) {
				float binMin = -99999.0f;
				lss >> binMin;
				oppositeSelectBinMins.push_back(binMin);
			}
		}
		else if (var == "oppositeSelectBinMaxes") {
			while (!lss.eof()) {
				float binMax = 99999.0f;
				lss >> binMax;
				oppositeSelectBinMaxes.push_back(binMax);
			}
		}
		else if (var == "randomizeSelectOptimizationStartPoint")
			lss >> randomizeSelectOptimizationStartPoint;
		else if (var == "testUseSampleFile")
			lss >> testUseSampleFile;
		else if (var == "testBegin")
			lss >> testBegin;
		else if (var == "testNumIOs")
			lss >> testNumIOs;
		else if (var == "discardSamples")
			lss >> discardSamples;
		else if (var == "testExplicitFile")
			lss >> testExplicitFile;
		else if (var == "selecttestfile")
			lss >> selecttestfile;
		else if (var == "selectionEvaluationType")
			lss >> selectionEvaluationType;
		else if (var == "selectPerturbSigma")
			lss >> selectPerturbSigma;
	}
}

//Because I am very very lazy, the long set is stored in "trainset" and the short in "test set"
size_t readData(size_t begin, size_t numIOs) {
	if (numIOs > 0)
		std::cout << "Reading " << numIOs << " samples from data set: ";
	else
		std::cout << "Reading all samples from data set: ";

	auto readstart = std::chrono::high_resolution_clock::now();
	size_t totalSamples;
	totalSamples = readTwoPriceTrainSet(selecttestfile, begin, numIOs);

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

SelectionCriteria getRandomSelectionCriteria() {
	SelectionCriteria crit;

	crit.minSubnetSelect = rand() % numSubnets + 1;
	crit.oppositeMinSubnetSelect = rand() % numSubnets + 1;
	crit.testSelectBinMins.resize(numBins);
	crit.testSelectBinMaxes.resize(numBins);
	crit.oppositeSelectBinMins.resize(numBins);
	crit.oppositeSelectBinMaxes.resize(numBins);
	for (size_t i = 0; i < numBins; i++) {
		int criteriaMin = -20;
		int criteriaMax = 125;
		if (rand() % 2 == 0) {
			crit.testSelectBinMins[i] = (float)(rand() % (criteriaMax - criteriaMin) + criteriaMin);
			crit.oppositeSelectBinMins[i] = (float)(rand() % (criteriaMax - criteriaMin) + criteriaMin);
			crit.testSelectBinMaxes[i] = rand() % (criteriaMax - (int)crit.testSelectBinMins[i]) + crit.testSelectBinMins[i];
			crit.oppositeSelectBinMaxes[i] = rand() % (criteriaMax - (int)crit.oppositeSelectBinMins[i]) + crit.oppositeSelectBinMins[i];
		}
		else {
			crit.testSelectBinMaxes[i] = (float)(rand() % (criteriaMax - criteriaMin) - criteriaMin);
			crit.oppositeSelectBinMaxes[i] = (float)(rand() % (criteriaMax - criteriaMin) - criteriaMin);
			crit.testSelectBinMins[i] = -(rand() % ((int)crit.testSelectBinMaxes[i] - criteriaMin)) + crit.testSelectBinMaxes[i];
			crit.oppositeSelectBinMins[i] = -(rand() % ((int)crit.oppositeSelectBinMaxes[i] - criteriaMin)) + crit.oppositeSelectBinMaxes[i];
		}
	}

	return crit;
}

SelectionCriteria perturbSelectionCriteria(SelectionCriteria crit) {
	size_t r = rand() % 4;
	if (r == 0 && crit.minSubnetSelect > 1)
		crit.minSubnetSelect--;
	else if (r == 1 && crit.minSubnetSelect < numSubnets)
		crit.minSubnetSelect++;

	r = rand() % 4;
	if (r == 0 && crit.oppositeMinSubnetSelect > 0)
		crit.oppositeMinSubnetSelect--;
	else if (r == 1 && crit.oppositeMinSubnetSelect < numSubnets)
		crit.oppositeMinSubnetSelect++;

	std::default_random_engine generator(rand());
	std::normal_distribution<double> distribution(0.0, selectPerturbSigma);

	for (size_t i = 0; i < numBins; i++) {
		crit.testSelectBinMins[i] += (float)distribution(generator);
		crit.testSelectBinMaxes[i] += (float)distribution(generator);
		crit.oppositeSelectBinMins[i] += (float)distribution(generator);
		crit.oppositeSelectBinMaxes[i] += (float)distribution(generator);
	}

	return crit;
}

void printSelectionCriteria(SelectionCriteria crit) {
	std::cout << "Min Subnets: T: " << crit.minSubnetSelect << " O: " << crit.oppositeMinSubnetSelect << std::endl;
	std::cout << "Test Bin Range: ";
	for (size_t i = 0; i < numBins; i++) {
		std::cout << "[" << crit.testSelectBinMins[i] << "," << crit.testSelectBinMaxes[i] << "] ";
	}
	std::cout << std::endl;
	std::cout << "Opposite Bin Range: ";
	for (size_t i = 0; i < numBins; i++) {
		std::cout << "[" << crit.oppositeSelectBinMins[i] << "," << crit.oppositeSelectBinMaxes[i] << "] ";
	}
	std::cout << std::endl;
}

void saveSelectionCriteria(SelectionCriteria crit) {
	std::stringstream css;
	css << savestring << savename << "sel";
	std::ofstream oss(css.str().c_str());

	oss << crit.minSubnetSelect << " " << crit.oppositeMinSubnetSelect << std::endl;
	for (size_t i = 0; i < numBins; i++) {
		oss << crit.testSelectBinMins[i] << " ";
	}
	oss << std::endl;
	for (size_t i = 0; i < numBins; i++) {
		oss << crit.testSelectBinMaxes[i] << " ";
	}
	oss << std::endl;
	for (size_t i = 0; i < numBins; i++) {
		oss << crit.oppositeSelectBinMins[i] << " ";
	}
	oss << std::endl;
	for (size_t i = 0; i < numBins; i++) {
		oss << crit.oppositeSelectBinMaxes[i] << " ";
	}
	oss << std::endl;
}

bool loadSelectionCriteria(SelectionCriteria* crit) {
	std::stringstream css;
	css << savestring << savename << "sel";

	if (!PathFileExists(css.str().c_str())) {
		std::cout << "No criteria file found" << std::endl;
		return false;
	}

	std::ifstream iss(css.str().c_str());

	iss >> crit->minSubnetSelect >> crit->oppositeMinSubnetSelect;
	crit->testSelectBinMins.resize(numBins);
	crit->testSelectBinMaxes.resize(numBins);
	crit->oppositeSelectBinMins.resize(numBins);
	crit->oppositeSelectBinMaxes.resize(numBins);
	for (size_t i = 0; i < numBins; i++) {
		iss >> crit->testSelectBinMins[i];
	}
	for (size_t i = 0; i < numBins; i++) {
		iss >> crit->testSelectBinMaxes[i];
	}
	for (size_t i = 0; i < numBins; i++) {
		iss >> crit->oppositeSelectBinMins[i];
	}
	for (size_t i = 0; i < numBins; i++) {
		iss >> crit->oppositeSelectBinMaxes[i];
	}
	return true;
}
