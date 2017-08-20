#include "stockrun.cuh"

#ifdef LOCAL
#define datastring "rawdata/"
#define savestring "saveweights/"
#else
#define datastring "../stockproj/rawdata/"
#define savestring "../stockproj/saveweights/"
#endif

float stockcombineBinPositiveOutput;
float stockcombineBinNegativeOutput;

std::vector<std::vector<std::vector<std::vector<float>>>> subnetResults;	//[dataset.size(), numCombineCVSets, 2*numCombineSubnets, numBins]

bool testExplicitFile = false;
size_t testBegin = 1;
size_t testNumIOs = 0;

bool testUseSampleFile = false;
bool discardSamples = false;

void loadLocalParameters(std::string parName);

size_t numCombineCVSets = 1;
size_t numCombineSubnets = 1;

std::vector<std::vector<LayerCollection>> longsubnets;	//[numCombineCVSets, numCombineSubnets]
std::vector<std::vector<LayerCollection>> shortsubnets;	//[numCombineCVSets, numCombineSubnets]
std::vector<std::vector<std::vector<float>>> scales;	//[numCombineCVSets, 2*numCombineSubnets, numBins]
std::vector<std::vector<std::vector<float>>> means;
std::vector<std::vector<float>> longBinEdges;			//[numCombineCVSets, numBins]
std::vector<std::vector<float>> shortBinEdges;

bool combineOutputScaling = true;

size_t combineLongNetsSelectPerCV = 9;
size_t combineShortNetsSelectPerCV = 9;

bool combineWithMinBackupWeights = true;

size_t binToCombineOn = 0;

std::string combineTestFile = "combinetest";

bool combineMergeSubnetsByCV = true;

size_t readData(std::string fname, size_t begin, size_t numIOs);

void generateSubnetResults();
void evaluateMaxBinSelection(size_t bin, size_t numCVSetsToSelect, bool print);
void evaluateMaxBinSelectionMergedSubnets(size_t bin, size_t numSubnetsToSelect, bool print);
bool loadScales(std::vector<float>* means, std::vector<float>* scales, std::string fname);
void stockcombineLoadBinEdges(std::vector<float>* edges, std::string binEdgeFile);
void stockcombineGetBinnedOutput(float output, std::vector<float>* localBinEdges, std::vector<float>* bins);

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

	longBinEdges.resize(numCombineCVSets);
	shortBinEdges.resize(numCombineCVSets);
	
	std::cout << "Loading bin edge files: ";
	for (size_t i = 0; i < numCombineCVSets; i++) {
		std::stringstream longbss;
		longbss << binEdgeFile << "long" << i + 1;
		stockcombineLoadBinEdges(&longBinEdges[i], longbss.str());
		std::stringstream shortbss;
		shortbss << binEdgeFile << "short" << i + 1;
		stockcombineLoadBinEdges(&shortBinEdges[i], shortbss.str());
	}
	std::cout << "Done." << std::endl;

	longsubnets.resize(numCombineCVSets);
	shortsubnets.resize(numCombineCVSets);
	for (size_t i = 0; i < numCombineCVSets; i++) {
		longsubnets[i].resize(numCombineSubnets);
		shortsubnets[i].resize(numCombineSubnets);
	}
	scales.resize(numCombineCVSets);
	means.resize(numCombineCVSets);
	numBins = longBinEdges[0].size() + 1;	//assume they're all the same size because I am very lazy.
	for (size_t i = 0; i < numCombineCVSets; i++) {
		scales[i].resize(2 * numCombineSubnets);
		means[i].resize(2 * numCombineSubnets);
		for (size_t j = 0; j < 2 * numCombineSubnets; j++) {
			std::vector<float>* edges;
			if (j < numCombineSubnets)
				edges = &longBinEdges[i];
			else
				edges = &shortBinEdges[i];
			if (edges->size() + 1 != numBins) {
				std::cout << "All binEdges files must have the same number of bins" << std::endl;
				throw std::runtime_error("All binEdges files must have the same number of bins");
			}
			scales[i][j].resize(numBins);
			means[i][j].resize(numBins);
		}
	}
	stockcombineBinPositiveOutput = BIN_POSITIVE_OUTPUT;
	stockcombineBinNegativeOutput = -stockcombineBinPositiveOutput / (numBins - 1);

	readData(combineTestFile, testBegin, testNumIOs);

	std::cout << "Loading subnets: ";
	for (size_t cv = 0; cv < numCombineCVSets || numCombineCVSets == 0; cv++) {
		for (size_t sub = 0; sub < numCombineSubnets; sub++) {
			longsubnets[cv][sub] = createLayerCollection(0, getLCType());
			initializeLayers(&longsubnets[cv][sub]);

			std::stringstream wss;
			if (combineWithMinBackupWeights)
				wss << "backup/";
			wss << savename;
			wss << "long";
			if (numCombineCVSets != 0)
				wss << cv + 1 << "-";
			wss << sub + 1;
			if (combineWithMinBackupWeights)
				wss << "Min";
			if (!loadWeights(longsubnets[cv][sub], wss.str().c_str())) {
				std::cout << "couldn't find long weights file, CV:" << cv + 1 << " Sub: " << sub + 1 << std::endl;
#ifdef LOCAL
				system("pause");
#endif
				return 0;
			}
			if (combineOutputScaling && !loadScales(&means[cv][sub], &scales[cv][sub], wss.str().c_str())) {
				std::cout << "Couldn't find long scales file #" << cv + 1 << "-" << sub+1 << std::endl;
#ifdef LOCAL
				system("pause");
#endif
				return 0;
			}

			shortsubnets[cv][sub] = createLayerCollection(0, getLCType());
			initializeLayers(&shortsubnets[cv][sub]);

			std::stringstream sss;
			if (combineWithMinBackupWeights)
				sss << "backup/";
			sss << savename;
			sss << "short";
			if (numCombineCVSets != 0)
				sss << cv + 1 << "-";
			sss << sub + 1;
			if (combineWithMinBackupWeights)
				sss << "Min";
			if (!loadWeights(shortsubnets[cv][sub], sss.str().c_str())) {
				std::cout << "couldn't find short weights file, CV:" << cv + 1 << " Sub: " << sub + 1 << std::endl;
#ifdef LOCAL
				system("pause");
#endif
				return 0;
			}
			if (combineOutputScaling && !loadScales(&means[cv][sub+numCombineSubnets], &scales[cv][sub+numCombineSubnets], wss.str().c_str())) {
				std::cout << "Couldn't find short scales file #" << cv + 1 << "-" << sub+1 << std::endl;
#ifdef LOCAL
				system("pause");
#endif
				return 0;
			}
		}
	}
	std::cout << "Done." << std::endl;

	std::cout << "Generating subnet results:";
	auto genstart = std::chrono::high_resolution_clock::now();
	generateSubnetResults();
	auto genelapsed = std::chrono::high_resolution_clock::now() - genstart;
	long long gentime = std::chrono::duration_cast<std::chrono::microseconds>(genelapsed).count();
	std::cout << " (" << gentime / 1000000 << " s)" << std::endl;

	if (combineMergeSubnetsByCV) {
		std::cout << "Printing results by required number of agreeing CV sets." << std::endl;
		for (size_t i = 0; i <= numCombineCVSets; i++) {
			evaluateMaxBinSelection(binToCombineOn, i, true);
		}
	}
	else {
		std::cout << "Printing results by required number of agreeing subnets." << std::endl;
		for (size_t i = 0; i <= numCombineCVSets*numCombineSubnets; i++) {
			evaluateMaxBinSelectionMergedSubnets(binToCombineOn, i, true);
		}
	}

#ifdef LOCAL
	system("pause");
#endif
}

//long outputs are stored in "correctoutput" and short in "secondaryoutput"
size_t readData(std::string fname, size_t begin, size_t numIOs) {
	if (numIOs > 0)
		std::cout << "Reading " << numIOs << " samples from data set " << fname << ": ";
	else
		std::cout << "Reading all samples from data set " << fname << ": ";

	auto readstart = std::chrono::high_resolution_clock::now();
	size_t totalSamples;
	totalSamples = readTwoPriceTrainSet(fname, begin, numIOs);

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

void generateSubnetResults() {
	std::vector<std::vector<float*>> d_inputs(numCombineCVSets, std::vector<float*>(2*numCombineSubnets));
	std::vector<std::vector<float*>> h_output(numCombineCVSets, std::vector<float*>(2*numCombineSubnets));
	std::vector<std::vector<float*>> d_output(numCombineCVSets, std::vector<float*>(2*numCombineSubnets));
	
	for (size_t cv = 0; cv < numCombineCVSets; cv++) {
		for (size_t sub = 0; sub < 2*numCombineSubnets; sub++) {
			LayerCollection* layers;
			size_t subnetPos = sub % numCombineSubnets;
			if (sub < numCombineSubnets)
				layers = &longsubnets[cv][subnetPos];
			else
				layers = &shortsubnets[cv][subnetPos];

			if (layers->numConvolutions > 0) {
				if (layers->convPars[0].numInputLocs != NUM_INPUTS || layers->convPars[0].numInputNeurons != 1)
					throw std::runtime_error("inputs to first layer don't match data set");
				d_inputs[cv][sub] = layers->convMat[0].inlayer;
			}
			else if (layers->numFixedNets > 0) {
				if (layers->fixedPars[0].numInputNeurons != NUM_INPUTS)
					throw std::runtime_error("inputs to first layer don't match data set");
				d_inputs[cv][sub] = layers->fixedMat[0].inlayer;
			}
			else
				throw std::runtime_error("tried to run on a network with no convolutions and no fixed networks");

			checkCudaErrors(cudaHostAlloc(&h_output[cv][sub], numBins*sizeof(float), cudaHostAllocMapped));
			checkCudaErrors(cudaHostGetDevicePointer(&d_output[cv][sub], h_output[cv][sub], 0));

			disableDropout();
			generateDropoutMask(layers);
		}
	}

	cudaStream_t mainStream = 0;

	std::vector<IOPair>* dataset = getTrainSet();
	subnetResults.resize(dataset->size());
	for (size_t i = 0; i < subnetResults.size(); i++) {
		subnetResults[i].resize(numCombineCVSets);
		for (size_t j = 0; j < subnetResults[i].size(); j++) {
			subnetResults[i][j].resize(2*numCombineSubnets);
			for (size_t k = 0; k < subnetResults[i][j].size(); k++) {
				subnetResults[i][j][k].resize(numBins);
			}
		}
	}

	std::vector<float> correctoutput;
	for (size_t i = 0; i < dataset->size(); i++) {
		for (size_t cv = 0; cv < numCombineCVSets; cv++) {
			for (size_t sub = 0; sub < 2 * numCombineSubnets; sub++) {
				LayerCollection* layers;
				size_t subnetPos = sub % numCombineSubnets;
				if (sub < numCombineSubnets)
					layers = &longsubnets[cv][subnetPos];
				else
					layers = &shortsubnets[cv][subnetPos];

				checkCudaErrors(cudaMemcpyAsync(d_inputs[cv][sub], &(*dataset)[i].inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice, mainStream));

				calculate((*layers), mainStream);

				if (sub < numCombineSubnets)
					stockcombineGetBinnedOutput((*dataset)[i].correctoutput, &longBinEdges[cv], &correctoutput);
				else
					stockcombineGetBinnedOutput((*dataset)[i].secondaryoutput, &shortBinEdges[cv], &correctoutput);
				checkCudaErrors(cudaMemcpyAsync(layers->correctoutput, &correctoutput[0], numBins*sizeof(float), cudaMemcpyHostToDevice, mainStream));

				calculateOutputError << <1, numBins, 0, mainStream >> >(layers->d_fixedMat[layers->numFixedNets - 1], layers->stepfactor, layers->correctoutput, d_output[cv][sub]);
			}
		}

		checkCudaErrors(cudaDeviceSynchronize());
		for (size_t cv = 0; cv < numCombineCVSets; cv++) {
			for (size_t sub = 0; sub < 2 * numCombineSubnets; sub++) {
				for (size_t j = 0; j < numBins; j++) {
					if (combineOutputScaling)
						subnetResults[i][cv][sub][j] = scales[cv][sub][j] * (h_output[cv][sub][j] - means[cv][sub][j]);
					else
						subnetResults[i][cv][sub][j] = h_output[cv][sub][j];
				}
			}
		}
	}
}

void evaluateMaxBinSelection(size_t bin, size_t numCVSetsToSelect, bool print) {
	std::vector<IOPair>* dataset = getTrainSet();
	float longProfit = 0;
	float shortProfit = 0;
	size_t numLongTrades = 0;
	size_t numShortTrades = 0;
	std::vector<size_t> longDist(numBins);
	std::vector<size_t> shortDist(numBins);

	for (size_t i = 0; i < subnetResults.size(); i++) {
		size_t numLongCVSelected = 0;
		size_t numShortCVSelected = 0;
		for (size_t cv = 0; cv < numCombineCVSets; cv++) {
			size_t numLongTestSelected = 0;
			size_t numShortTestSelected = 0;
			for (size_t j = 0; j < 2 * numCombineSubnets; j++) {
				float maxBinWeight = 0.0f;
				size_t maxBin = 0;
				for (size_t k = 0; k < numBins; k++) {
					if (subnetResults[i][cv][j][k] > maxBinWeight) {
						maxBinWeight = subnetResults[i][cv][j][k];
						maxBin = k;
					}
				}

				if (j < numCombineSubnets) { //LONG
					if (maxBinWeight > 0.0f && maxBin == bin) {
						numLongTestSelected++;
					}
				}
				else {
					if (maxBinWeight > 0.0f && maxBin == bin)
						numShortTestSelected++;
				}
			}
			if (numLongTestSelected >= combineLongNetsSelectPerCV) {
				numLongCVSelected++;
			}
			if (numShortTestSelected >= combineShortNetsSelectPerCV) {
				numShortCVSelected++;
			}
		}
		if (numLongCVSelected >= numCVSetsToSelect) {
			longProfit += (*dataset)[i].correctoutput;
			numLongTrades++;
			/*
			size_t binPos = 0;
			for (size_t j = 0; j < numBins; j++) {
				if ((*dataset)[i].correctbins[j] == BIN_POSITIVE_OUTPUT) {
					binPos = j;
					break;
				}
			}
			longDist[binPos]++;
			*/
		}
		if (numShortCVSelected >= numCVSetsToSelect) {
			shortProfit += (*dataset)[i].secondaryoutput;
			numShortTrades++;
			/*
			size_t binPos = 0;
			for (size_t j = 0; j < numBins; j++) {
				if ((*dataset)[i].secondarybins[j] == BIN_POSITIVE_OUTPUT) {
					binPos = j;
					break;
				}
			}
			shortDist[binPos]++;
			*/
		}
	}

	if (print) {
		std::cout << "Bin #" << bin << " with " << numCVSetsToSelect << " CVs with " << combineLongNetsSelectPerCV << "/" << combineShortNetsSelectPerCV << " nets required. ";
		std::cout << "Profits: L: " << longProfit << " (/" << numLongTrades << "=" << longProfit / numLongTrades << ") S: " << shortProfit << " (/" << numShortTrades << "=" << shortProfit / numShortTrades << ")" << " Total: " << longProfit + shortProfit << " (/" << numLongTrades + numShortTrades << "=" << (longProfit + shortProfit) / (numLongTrades + numShortTrades) << ")" << std::endl;
		/*
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
		*/
	}
}

void evaluateMaxBinSelectionMergedSubnets(size_t bin, size_t numSubnetsToSelect, bool print) {
	std::vector<IOPair>* dataset = getTrainSet();
	float longProfit = 0;
	float shortProfit = 0;
	size_t numLongTrades = 0;
	size_t numShortTrades = 0;
	std::vector<size_t> longDist(numBins);
	std::vector<size_t> shortDist(numBins);

	for (size_t i = 0; i < subnetResults.size(); i++) {
		size_t numLongNetsSelected = 0;
		size_t numShortNetsSelected = 0;
		for (size_t cv = 0; cv < numCombineCVSets; cv++) {
			for (size_t j = 0; j < 2 * numCombineSubnets; j++) {
				float maxBinWeight = 0.0f;
				size_t maxBin = 0;
				for (size_t k = 0; k < numBins; k++) {
					if (subnetResults[i][cv][j][k] > maxBinWeight) {
						maxBinWeight = subnetResults[i][cv][j][k];
						maxBin = k;
					}
				}

				if (j < numCombineSubnets) { //LONG
					if (maxBinWeight > 0.0f && maxBin == bin) {
						numLongNetsSelected++;
					}
				}
				else {
					if (maxBinWeight > 0.0f && maxBin == bin)
						numShortNetsSelected++;
				}
			}
		}
		if (numLongNetsSelected >= numSubnetsToSelect) {
			longProfit += (*dataset)[i].correctoutput;
			numLongTrades++;
			/*
			size_t binPos = 0;
			for (size_t j = 0; j < numBins; j++) {
				if ((*dataset)[i].correctbins[j] == BIN_POSITIVE_OUTPUT) {
					binPos = j;
					break;
				}
			}
			longDist[binPos]++;
			*/
		}
		if (numShortNetsSelected >= numSubnetsToSelect) {
			shortProfit += (*dataset)[i].secondaryoutput;
			numShortTrades++;
			/*
			size_t binPos = 0;
			for (size_t j = 0; j < numBins; j++) {
				if ((*dataset)[i].secondarybins[j] == BIN_POSITIVE_OUTPUT) {
					binPos = j;
					break;
				}
			}
			shortDist[binPos]++;
			*/
		}
	}

	if (print) {
		std::cout << "Bin #" << bin << " with " << numSubnetsToSelect << " nets required. ";
		std::cout << "Profits: L: " << longProfit << " (/" << numLongTrades << "=" << longProfit / numLongTrades << ") S: " << shortProfit << " (/" << numShortTrades << "=" << shortProfit / numShortTrades << ")" << " Total: " << longProfit + shortProfit << " (/" << numLongTrades + numShortTrades << "=" << (longProfit + shortProfit) / (numLongTrades + numShortTrades) << ")" << std::endl;
		/*
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
		*/
	}
}

bool loadScales(std::vector<float>* means, std::vector<float>* scales, std::string fname) {
	scales->resize(numBins);
	means->resize(numBins);

	std::stringstream fss;
	fss << savestring << fname << "scales";

	if (!PathFileExists(fss.str().c_str())) {
		std::cout << "No scales file found for file " << fname << std::endl;
		return false;
	}

	std::ifstream infile(fss.str().c_str());
	
	for (size_t i = 0; i < numBins; i++) {
		infile >> (*means)[i];
	}
	for (size_t i = 0; i < numBins; i++) {
		infile >> (*scales)[i];
	}
	return true;
}

void loadLocalParameters(std::string parName) {
	std::ifstream infile(parName.c_str());
	std::string line;
	while (getline(infile, line)) {
		std::stringstream lss(line);
		std::string var;
		lss >> var;

		if (var == "numCombineCVSets")
			lss >> numCombineCVSets;
		else if (var == "numCombineSubnets")
			lss >> numCombineSubnets;
		else if (var == "combineLongNetsSelectPerCV")
			lss >> combineLongNetsSelectPerCV;
		else if (var == "combineShortNetsSelectPerCV")
			lss >> combineShortNetsSelectPerCV;
		else if (var == "combineWithMinBackupWeights")
			lss >> combineWithMinBackupWeights;
		else if (var == "binToCombineOn")
			lss >> binToCombineOn;
		else if (var == "combineTestFile")
			lss >> combineTestFile;
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
		else if (var == "combineOutputScaling")
			lss >> combineOutputScaling;
		else if (var == "combineMergeSubnetsByCV")
			lss >> combineMergeSubnetsByCV;
	}
}

void stockcombineLoadBinEdges(std::vector<float>* edges, std::string binEdgeFile) {
	edges->clear();
	std::stringstream ss;
	ss << datastring << binEdgeFile;
	std::ifstream binfile(ss.str());
	if (!binfile.is_open()) {
		std::cout << "Couldn't load binEdge file " << ss.str() << ", proceeding without binEdges." << std::endl;
		return;
	}
	else
		std::cout << "Loaded binEdge file " << ss.str() << std::endl;

	float dum;
	while (binfile >> dum)
		edges->push_back(dum);
}

void stockcombineGetBinnedOutput(float output, std::vector<float>* localBinEdges, std::vector<float>* bins) {
	if (numBins == 0 || localBinEdges->size() == 0 || localBinEdges->size() != numBins - 1) {
		throw std::runtime_error("Invalid numBins or binEdges");
	}

	bins->resize(numBins);
	
	for (size_t i = 1; i < numBins-1; i++) {
		if (output >= (*localBinEdges)[i-1] && output < (*localBinEdges)[i])
			(*bins)[i] = stockcombineBinPositiveOutput;
		else
			(*bins)[i] = stockcombineBinNegativeOutput;
	}
	if (output < (*localBinEdges)[0])
		(*bins)[0] = stockcombineBinPositiveOutput;
	else
		(*bins)[0] = stockcombineBinNegativeOutput;

	if (numBins >= 2 && output > (*localBinEdges)[numBins - 2])
		(*bins)[numBins - 1] = stockcombineBinPositiveOutput;
	else
		(*bins)[numBins - 1] = stockcombineBinNegativeOutput;
}
