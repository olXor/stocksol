#include <stockrun.cuh>
#include <windows.h>
#include <deque>

#ifdef LOCAL
#define datastring "rawdata/"
#define savestring "saveweights/"
#else
#define datastring "../stockproj/rawdata/"
#define savestring "../stockproj/saveweights/"
#endif

struct SelectionCriteria {
	size_t minSubnetSelect;
	size_t oppositeMinSubnetSelect;
	std::vector<float> testSelectBinMins;
	std::vector<float> testSelectBinMaxes;
	std::vector<float> oppositeSelectBinMins;
	std::vector<float> oppositeSelectBinMaxes;
};

std::vector<std::vector<LayerCollection>> longsubnets;
std::vector<std::vector<LayerCollection>> shortsubnets;

std::vector<std::vector<float*>> d_inputs;
std::vector<std::vector<float*>> h_outputs;

std::deque<float> inputqueue;

size_t numCombineCVSets = 1;
size_t numCombineSubnets = 1;

size_t combineNetsSelectPerCV = 7;
size_t finalCombineCVsToSelect = 5;

bool combineWithMinBackupWeights = true;

size_t binToCombineOn = 4;

HANDLE createPipe(std::string name);
void connectPipe(HANDLE pipe);
bool sendString(HANDLE pipe, const char* data);
bool readString(HANDLE pipe, char* buffer, int size);
bool evaluateInput(HANDLE pipe);
void createSubnetInputOutput();
size_t selectTrade(std::vector<float> inputs);
bool loadSelectionCriteria(SelectionCriteria* crit);
void loadLocalParameters(std::string parName);

int main() {
	srand((size_t)time(NULL));
#ifdef LOCAL
	loadLocalParameters("pars.cfg");
	loadParameters("pars.cfg");
#else
	loadLocalParameters("../stockproj/pars.cfg");
	loadParameters("../stockproj/pars.cfg");
#endif
	setStrings(datastring, savestring);

	longsubnets.resize(numCombineCVSets);
	shortsubnets.resize(numCombineCVSets);
	for (size_t i = 0; i < numCombineCVSets; i++) {
		longsubnets[i].resize(numCombineSubnets);
		shortsubnets[i].resize(numCombineSubnets);
	}

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
		}
	}
	std::cout << "Done." << std::endl;
	createSubnetInputOutput();

	while (true) {
		std::cout << "Creating MT4 pipe: ";
		HANDLE pipe = createPipe("mt4pipe");
		std::cout << "done." << std::endl;

		std::cout << "Waiting for pipe to connect: ";
		connectPipe(pipe);
		std::cout << "done." << std::endl;
		//SelectionCriteria crit;
		//loadSelectionCriteria(&crit);

		while (evaluateInput(pipe)) {}
		CloseHandle(pipe);
	}
}

HANDLE createPipe(std::string name) {
	std::string pname = "\\\\.\\pipe\\" + name;
	HANDLE pipe = CreateNamedPipe(
		pname.c_str(),
		PIPE_ACCESS_DUPLEX,
		PIPE_TYPE_BYTE,
		1, 0, 0, 0, NULL);

	if (pipe == NULL || pipe == INVALID_HANDLE_VALUE) {
		std::cout << "Failed to open pipe" << std::endl;
		throw std::runtime_error("Failed to open pipe");
	}
	return pipe;
}

void connectPipe(HANDLE pipe) {
	bool result = ConnectNamedPipe(pipe, NULL) != 0;
	if (!result) {
		std::cout << "Failed to make connection" << std::endl;
		throw std::runtime_error("Failed to make connection");
	}
}

bool sendString(HANDLE pipe, const char* data) {
	DWORD numBytesWritten = 0;
	return WriteFile(pipe, data, strlen(data) * sizeof(char), &numBytesWritten, NULL) != 0;
}

bool readString(HANDLE pipe, char* buffer, int size) {
	DWORD numBytesWritten = 0;
	return ReadFile(pipe, buffer, size*sizeof(char), &numBytesWritten, NULL) != 0;
}

bool evaluateInput(HANDLE pipe) {
	size_t size = 30 * NUM_INPUTS;
	char* buffer = new char[size];
	//std::vector<float> inputs;
	
	if (!readString(pipe, buffer, size)) return false;

	std::stringstream bss;
	bss << buffer;
	float in;
	if (!(bss >> in)) {
		std::cout << "Received invalid string from pipe" << std::endl;
		throw std::runtime_error("Received invalid string from pipe");
	}

	inputqueue.push_back(in);
	while (inputqueue.size() > NUM_INPUTS)
		inputqueue.pop_front();
	std::vector<float> inputs;

	size_t output;
	if (inputqueue.size() == NUM_INPUTS) {
		for (float i : inputqueue) {
			inputs.push_back(i);
		}

		float maxinput = -999999;
		float mininput = 999999;
		for (size_t j = 0; j < NUM_INPUTS; j++) {
			if (inputs[j] > maxinput)
				maxinput = inputs[j];
			if (inputs[j] < mininput)
				mininput = inputs[j];
		}
		for (size_t j = 0; j<NUM_INPUTS; j++) {
			if (maxinput > mininput)
				inputs[j] = 2 * (inputs[j] - mininput) / (maxinput - mininput) - 1;
			else
				inputs[j] = 0;
		}

		output = selectTrade(inputs);
	}
	else
		output = 0;
	std::stringstream oss;
	oss << output;

	if (!sendString(pipe, oss.str().c_str())) return false;

	/*
	std::stringstream coutss;
	coutss << "Inputs: ";
	for (size_t i = 0; i < NUM_INPUTS; i++) {
		coutss << inputs[i] << " ";
	}
	coutss << " Output: " << output;
	std::cout << coutss.str() << std::endl;
	*/

	delete [] buffer;

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
		else if (var == "combineNetsSelectPerCV")
			lss >> combineNetsSelectPerCV;
		else if (var == "combineWithMinBackupWeights")
			lss >> combineWithMinBackupWeights;
		else if (var == "binToCombineOn")
			lss >> binToCombineOn;
		else if (var == "finalCombineCVsToSelect")
			lss >> finalCombineCVsToSelect;
	}
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

void createSubnetInputOutput() {
	d_inputs.resize(numCombineCVSets);
	h_outputs.resize(numCombineCVSets);
	
	for (size_t cv = 0; cv < numCombineCVSets; cv++) {
		d_inputs[cv].resize(2 * numCombineSubnets);
		h_outputs[cv].resize(2 * numCombineSubnets);
		for (size_t sub = 0; sub < 2 * numCombineSubnets; sub++) {
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

			h_outputs[cv][sub] = new float[numBins];
		}
	}
}

size_t selectTrade(std::vector<float> inputs) {
	for (size_t cv = 0; cv < numCombineCVSets; cv++) {
		for (size_t sub = 0; sub < 2 * numCombineSubnets; sub++) {
			LayerCollection layers;
			size_t subnetPos = sub % numCombineSubnets;
			if (sub < numCombineSubnets)
				layers = longsubnets[cv][subnetPos];
			else
				layers = shortsubnets[cv][subnetPos];

			disableDropout();
			generateDropoutMask(&layers);

			checkCudaErrors(cudaMemcpyAsync(d_inputs[cv][sub], &inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice));

			calculate(layers);

			checkCudaErrors(cudaMemcpyAsync(h_outputs[cv][sub], layers.fixedMat[layers.numFixedNets - 1].outlayer, numBins*sizeof(float), cudaMemcpyDeviceToHost));
		}
	}
	checkCudaErrors(cudaDeviceSynchronize());

	bool tradeLong = false;
	bool tradeShort = false;

	size_t numLongCVSelected = 0;
	size_t numShortCVSelected = 0;
	for (size_t cv = 0; cv < numCombineCVSets; cv++) {
		size_t numLongTestSelected = 0;
		size_t numShortTestSelected = 0;
		for (size_t j = 0; j < 2 * numCombineSubnets; j++) {
			float maxBinWeight = 0.0f;
			size_t maxBin = 0;
			for (size_t k = 0; k < numBins; k++) {
				if (h_outputs[cv][j][k] > maxBinWeight) {
					maxBinWeight = h_outputs[cv][j][k];
					maxBin = k;
				}
			}

			if (j < numCombineSubnets) { //LONG
				if (maxBinWeight > 0.0f && maxBin == binToCombineOn) {
					numLongTestSelected++;
				}
			}
			else {
				if (maxBinWeight > 0.0f && maxBin == binToCombineOn)
					numShortTestSelected++;
			}
		}
		if (numLongTestSelected >= combineNetsSelectPerCV) {
			numLongCVSelected++;
		}
		if (numShortTestSelected >= combineNetsSelectPerCV) {
			numShortCVSelected++;
		}
	}
	if (numLongCVSelected >= finalCombineCVsToSelect) {
		tradeLong = true;
	}
	if (numShortCVSelected >= finalCombineCVsToSelect) {
		tradeShort = true;
	}

	if (!(tradeLong || tradeShort))
		return 0;
	else if (tradeLong && !tradeShort)
		return 1;
	else if (!tradeLong && tradeShort)
		return 2;
	else if (tradeLong && tradeShort)
		return 3;
	return 0;
}