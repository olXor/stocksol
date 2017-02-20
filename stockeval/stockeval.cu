#include "stockrun.cuh"
#include <random>

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

std::string evalTestFile = "testset";

std::vector<LayerCollection> longsubnets;
std::vector<LayerCollection> shortsubnets;

size_t numEvalSubnets = 1;
size_t numEvalCVs = 0;
bool evalOnMinBackupWeights = true;

bool evalExplicitFile = false;
bool evalExplicitFileLong = true;

void loadLocalParameters(std::string parName);
size_t readData(std::string fname, size_t begin, size_t numIOs);
void generateErrorMatrix(std::vector<std::vector<size_t>>* longTotalErrorMatrix, std::vector<std::vector<size_t>>* shortTotalErrorMatrix);
size_t readEvalExplicitTrainset(std::string learnsetname, bool longTrade);

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

	longsubnets.resize(numEvalSubnets);
	shortsubnets.resize(numEvalSubnets);

	std::vector<std::vector<size_t>> longTotalErrorMatrix(numBins, std::vector<size_t>(numBins));
	std::vector<std::vector<size_t>> shortTotalErrorMatrix(numBins, std::vector<size_t>(numBins));
	for (size_t cv = 0; cv < numEvalCVs || numEvalCVs == 0; cv++) {
		std::string fname = evalTestFile;
		if (numEvalCVs != 0) {
			std::stringstream fss;
			fss << evalTestFile << cv + 1;
			fname = fss.str();
		}
		readData(fname, testBegin, testNumIOs);
		std::cout << "Loading subnets for CV set " << cv + 1 << ": ";
		for (size_t sub = 0; sub < numEvalSubnets; sub++) {
			longsubnets[sub] = createLayerCollection(0, getLCType());
			initializeLayers(&longsubnets[sub]);

			std::stringstream wss;
			if (evalOnMinBackupWeights)
				wss << "backup/";
			wss << savename;
			wss << "long";
			if (numEvalCVs != 0)
				wss << cv + 1 << "-";
			wss << sub + 1;
			if (evalOnMinBackupWeights)
				wss << "Min";
			//std::cout << "Loading subnet " << wss.str().c_str() << std::endl;
			if (!loadWeights(longsubnets[sub], wss.str().c_str())) {
				std::cout << "couldn't find long weights file #" << sub + 1 << std::endl;
#ifdef LOCAL
				system("pause");
#endif
				return 0;
			}
			shortsubnets[sub] = createLayerCollection(0, getLCType());
			initializeLayers(&shortsubnets[sub]);

			std::stringstream sss;
			if (evalOnMinBackupWeights)
				sss << "backup/";
			sss << savename;
			sss << "short";
			if (numEvalCVs != 0)
				sss << cv + 1 << "-";
			sss << sub + 1;
			if (evalOnMinBackupWeights)
				sss << "Min";
			//std::cout << "Loading subnet " << sss.str().c_str() << std::endl;
			if (!loadWeights(shortsubnets[sub], sss.str().c_str())) {
				std::cout << "couldn't find short weights file #" << sub + 1 << std::endl;
#ifdef LOCAL
				system("pause");
#endif
				return 0;
			}
		}
		std::cout << "done" << std::endl;

		generateErrorMatrix(&longTotalErrorMatrix, &shortTotalErrorMatrix);
		std::cout << std::endl;
	}
	std::cout << "Totals: " << std::endl;
	std::cout << "Long - Short" << std::endl;
	for (size_t i = 0; i < numBins; i++) {
		size_t longtotal = 0;
		for (size_t j = 0; j < numBins; j++)
			longtotal += longTotalErrorMatrix[i][j];
		for (size_t j = 0; j < numBins; j++) {
			std::cout << 1.0f*longTotalErrorMatrix[i][j]/longtotal << " ";
		}
		std::cout << longtotal;
		std::cout << "          ";
		size_t shorttotal = 0;
		for (size_t j = 0; j < numBins; j++)
			shorttotal += shortTotalErrorMatrix[i][j];
		for (size_t j = 0; j < numBins; j++) {
			std::cout << 1.0f*shortTotalErrorMatrix[i][j]/shorttotal << " ";
		}
		std::cout << shorttotal;
		std::cout << std::endl;
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

	markTime();
	size_t totalSamples;
	if (!evalExplicitFile)
		totalSamples = readTwoPriceTrainSet(fname, begin, numIOs);
	else
		totalSamples = readEvalExplicitTrainset(fname, evalExplicitFileLong);

	std::cout << getTimeSinceMark() << " s" << std::endl;
	size_t numSamples;
	if (numIOs > 0)
		numSamples = min(numIOs, totalSamples);
	else
		numSamples = totalSamples;
	std::cout << numSamples << "/" << totalSamples << " samples loaded" << std::endl;
	return numSamples;
}

void generateErrorMatrix(std::vector<std::vector<size_t>>* longTotalErrorMatrix, std::vector<std::vector<size_t>>* shortTotalErrorMatrix) {
	std::cout << "Generating error matrix: ";
	markTime();
	std::vector<float*> d_inputs(2 * numEvalSubnets);
	std::vector<float*> h_output(2 * numEvalSubnets), d_output(2 * numEvalSubnets);
	for (size_t i = 0; i < 2 * numEvalSubnets; i++) {
		LayerCollection layers;
		size_t subnetPos = i % numEvalSubnets;
		if (i < numEvalSubnets) {
			if (evalExplicitFile && !evalExplicitFileLong)
				continue;
			layers = longsubnets[subnetPos];
		}
		else {
			if (evalExplicitFile && evalExplicitFileLong)
				continue;
			layers = shortsubnets[subnetPos];
		}

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
	std::vector<IOPair>* dataset = getTrainSet();
	std::vector<std::vector<size_t>> longErrorMatrix(numBins, std::vector<size_t>(numBins));
	std::vector<std::vector<size_t>> shortErrorMatrix(numBins, std::vector<size_t>(numBins));

	for (size_t i = 0; i < dataset->size(); i++) {
		for (size_t j = 0; j < 2 * numEvalSubnets; j++) {
			LayerCollection layers;
			size_t subnetPos = j % numEvalSubnets;
			if (j < numEvalSubnets) {
				if (evalExplicitFile && !evalExplicitFileLong)
					continue;
				layers = longsubnets[subnetPos];
			}
			else {
				if (evalExplicitFile && evalExplicitFileLong)
					continue;
				layers = shortsubnets[subnetPos];
			}

			checkCudaErrors(cudaMemcpyAsync(d_inputs[j], &(*dataset)[i].inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice));

			calculate(layers);

			checkCudaErrors(cudaMemcpyAsync(h_output[j], layers.fixedMat[layers.numFixedNets - 1].outlayer, numBins*sizeof(float), cudaMemcpyDeviceToHost));
		}
		checkCudaErrors(cudaDeviceSynchronize());
		for (size_t j = 0; j < 2 * numEvalSubnets; j++) {
			bool isLong = (j < numEvalSubnets);
			size_t binPos = 0;
			size_t corBinPos = 0;
			std::vector<float> corBins;
			if (isLong) {
				if (evalExplicitFile && !evalExplicitFileLong)
					continue;
				corBins = (*dataset)[i].correctbins;
			}
			else {
				if (evalExplicitFile && evalExplicitFileLong)
					continue;
				corBins = (*dataset)[i].secondarybins;
			}

			for (size_t k = 0; k < numBins; k++) {
				if (corBins[k] == BIN_POSITIVE_OUTPUT) {
					corBinPos = k;
					break;
				}
			}
			float maxBin = -9999;
			for (size_t k = 0; k < numBins; k++) {
				if (h_output[j][k] > maxBin) {
					maxBin = h_output[j][k];
					binPos = k;
				}
			}
			//---TEST
			/*
			std::cout << corBinPos << " " << binPos << " ";
			if (isLong)
				std::cout << (*dataset)[i].correctoutput << " ";
			else
				std::cout << (*dataset)[i].secondaryoutput << " ";
			for (size_t t = 0; t < numBins; t++)
				std::cout << h_output[j][t] << " ";
			std::cout << std::endl;
			*/
			//---TEST
			if (isLong) {
				longErrorMatrix[corBinPos][binPos]++;
				(*longTotalErrorMatrix)[corBinPos][binPos]++;
			}
			else {
				shortErrorMatrix[corBinPos][binPos]++;
				(*shortTotalErrorMatrix)[corBinPos][binPos]++;
			}
		}
	}
	std::cout << "Done. (" << getTimeSinceMark() << " s)" << std::endl;

	std::cout << "Long - Short" << std::endl;
	for (size_t i = 0; i < numBins; i++) {
		size_t longtotal = 0;
		for (size_t j = 0; j < numBins; j++)
			longtotal += longErrorMatrix[i][j];
		for (size_t j = 0; j < numBins; j++) {
			if (longtotal == 0)
				std::cout << "0" << " ";
			else
				std::cout << 1.0f*longErrorMatrix[i][j] / longtotal << " ";
		}
		std::cout << longtotal;
		std::cout << "          ";
		size_t shorttotal = 0;
		for (size_t j = 0; j < numBins; j++)
			shorttotal += shortErrorMatrix[i][j];
		for (size_t j = 0; j < numBins; j++) {
			if (shorttotal == 0)
				std::cout << "0" << " ";
			else
				std::cout << 1.0f*shortErrorMatrix[i][j] / shorttotal << " ";
		}
		std::cout << shorttotal;
		std::cout << std::endl;
	}
}

size_t readEvalExplicitTrainset(std::string learnsetname, bool longTrade) {
	std::vector<IOPair>*dataset = getTrainSet();

	(*dataset).clear();
	std::stringstream learnsetss;
	learnsetss << datastring << learnsetname;
	std::ifstream learnset(learnsetss.str().c_str());

	std::string line;
	std::string dum;
	size_t ionum = 0;
	while (getline(learnset, line)) {
		ionum++;

		IOPair io;
		io.samplenum = 0;
		io.inputs.resize(NUM_INPUTS);
		std::stringstream lss(line);
		if (lss.eof()) throw std::runtime_error("Invalid train set file!");

		float correctoutput;
		lss >> correctoutput;
		if (longTrade){
			io.correctoutput = correctoutput / OUTPUT_DIVISOR;
			io.correctbins = getBinnedOutput(correctoutput);
		}
		else {
			io.secondaryoutput = correctoutput / OUTPUT_DIVISOR;
			io.secondarybins = getBinnedOutput(correctoutput);
		}

		lss >> dum;		//"|"
		for (size_t i = 0; i < NUM_INPUTS; i++) {
			if (lss.eof()) throw std::runtime_error("Invalid train set file!");
			lss >> io.inputs[i];
		}
		(*dataset).push_back(io);
	}
	return ionum;
}

void loadLocalParameters(std::string parName) {
	std::ifstream infile(parName.c_str());
	std::string line;
	while (getline(infile, line)) {
		std::stringstream lss(line);
		std::string var;
		lss >> var;

		if (var == "evalTestFile")
			lss >> evalTestFile;
		else if (var == "numEvalSubnets")
			lss >> numEvalSubnets;
		else if (var == "numEvalCVs")
			lss >> numEvalCVs;
		else if (var == "evalOnMinBackupWeights")
			lss >> evalOnMinBackupWeights;
		else if (var == "evalExplicitFile")
			lss >> evalExplicitFile;
		else if (var == "evalExplicitFileLong")
			lss >> evalExplicitFileLong;
	}
}