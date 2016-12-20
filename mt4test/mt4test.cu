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

size_t numSubnets = 1;

std::vector<LayerCollection> longsubnets;
std::vector<LayerCollection> shortsubnets;

std::vector<float*> d_inputs;
std::vector<float*> h_outputs;

std::deque<float> inputqueue;

HANDLE createPipe(std::string name);
void connectPipe(HANDLE pipe);
bool sendString(HANDLE pipe, const char* data);
bool readString(HANDLE pipe, char* buffer, int size);
bool evaluateInput(HANDLE pipe, SelectionCriteria crit);
void createSubnetInputOutput();
size_t selectTrade(std::vector<float> inputs, SelectionCriteria crit);
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

	longsubnets.resize(numSubnets);
	shortsubnets.resize(numSubnets);
	for (size_t i = 0; i < numSubnets; i++) {
		longsubnets[i] = createLayerCollection(0, getLCType());
		initializeLayers(&longsubnets[i]);

		std::stringstream lss;
		lss << savename << "long" << i + 1;
		if (!loadWeights(longsubnets[i], lss.str().c_str())) {
			std::cout << "couldn't find long weights file #" << i + 1 << std::endl;
#ifdef LOCAL
			system("pause");
#endif
			return 0;
		}
		shortsubnets[i] = createLayerCollection(0, getLCType());
		initializeLayers(&shortsubnets[i]);

		std::stringstream sss;
		sss << savename << "short" << i + 1;
		if (!loadWeights(shortsubnets[i], sss.str().c_str())) {
			std::cout << "couldn't find short weights file #" << i + 1 << std::endl;
#ifdef LOCAL
			system("pause");
#endif
			return 0;
		}
	}
	createSubnetInputOutput();

	while (true) {
		std::cout << "Creating MT4 pipe: ";
		HANDLE pipe = createPipe("mt4pipe");
		std::cout << "done." << std::endl;

		std::cout << "Waiting for pipe to connect: ";
		connectPipe(pipe);
		std::cout << "done." << std::endl;
		SelectionCriteria crit;
		loadSelectionCriteria(&crit);

		while (evaluateInput(pipe, crit)) {}
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

bool evaluateInput(HANDLE pipe, SelectionCriteria crit) {
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

		output = selectTrade(inputs, crit);
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

		if (var == "numSubnets")
			lss >> numSubnets;
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
	d_inputs.resize(2 * numSubnets);
	h_outputs.resize(2 * numSubnets);
	for (size_t i = 0; i < 2 * numSubnets; i++) {
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

		h_outputs[i] = new float[numBins];
	}
}

size_t selectTrade(std::vector<float> inputs, SelectionCriteria crit) {
	for (size_t i = 0; i < 2*numSubnets; i++) {
		LayerCollection layers;
		size_t subnetPos = i % numSubnets;
		if (i < numSubnets)
			layers = longsubnets[subnetPos];
		else
			layers = shortsubnets[subnetPos];

		disableDropout();
		generateDropoutMask(&layers);

		checkCudaErrors(cudaMemcpyAsync(d_inputs[i], &inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice));

		calculate(layers);

		checkCudaErrors(cudaMemcpyAsync(h_outputs[i], layers.fixedMat[layers.numFixedNets - 1].outlayer, numBins*sizeof(float), cudaMemcpyDeviceToHost));
	}
	checkCudaErrors(cudaDeviceSynchronize());

	size_t numLongTestSelected = 0;
	size_t numLongOppositeSelected = 0;
	size_t numShortTestSelected = 0;
	size_t numShortOppositeSelected = 0;
	bool tradeLong = false;
	bool tradeShort = false;
	for (size_t j = 0; j < 2 * numSubnets; j++) {
		bool testSelected = true;
		bool oppositeSelected = true;
		for (size_t k = 0; k < numBins; k++) {
			if (h_outputs[j][k] < crit.testSelectBinMins[k] || h_outputs[j][k] > crit.testSelectBinMaxes[k])
				testSelected = false;
			if (h_outputs[j][k] < crit.oppositeSelectBinMins[k] || h_outputs[j][k] > crit.oppositeSelectBinMaxes[k])
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

	/*
	if (numLongTestSelected > 1 || numShortTestSelected > 1)
		std::cout << numLongTestSelected << " " << numShortTestSelected << " " << numLongOppositeSelected << " " << numShortOppositeSelected << std::endl;
		*/

	if (numLongTestSelected >= crit.minSubnetSelect && numLongOppositeSelected >= crit.oppositeMinSubnetSelect) {
		tradeLong = true;
	}
	if (numShortTestSelected >= crit.minSubnetSelect && numShortOppositeSelected >= crit.oppositeMinSubnetSelect) {
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