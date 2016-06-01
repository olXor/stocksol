#include <stockrun.cuh>
#include <windows.h>

#ifdef LOCAL
#define datastring "rawdata/"
#define savestring "saveweights/"
#else
#define datastring "../stockproj/rawdata/"
#define savestring "../stockproj/saveweights/"
#endif

HANDLE createPipe(std::string name);
void connectPipe(HANDLE pipe);
bool sendString(HANDLE pipe, const char* data);
bool readString(HANDLE pipe, char* buffer, int size);
bool evaluateInput(HANDLE pipe, LayerCollection layers);

int main() {
	srand((size_t)time(NULL));
#ifdef LOCAL
	//loadLocalParameters("pars.cfg");
	loadParameters("pars.cfg");
#else
	//loadLocalParameters("../stockproj/pars.cfg");
	loadParameters("../stockproj/pars.cfg");
#endif
	setStrings(datastring, savestring);

	LayerCollection layers = createLayerCollection();
	initializeLayers(&layers);
	loadWeights(layers, savename);

	while (true) {
		std::cout << "Creating MT4 pipe: ";
		HANDLE pipe = createPipe("mt4pipe");
		std::cout << "done." << std::endl;

		std::cout << "Waiting for pipe to connect: ";
		connectPipe(pipe);
		std::cout << "done." << std::endl;

		while (evaluateInput(pipe, layers)) {}
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

bool evaluateInput(HANDLE pipe, LayerCollection layers) {
	size_t size = 30 * NUM_INPUTS;
	char* buffer = new char[size];
	std::vector<float> inputs;
	
	if (!readString(pipe, buffer, size)) return false;
	std::stringstream bss;
	bss << buffer;
	for (size_t i = 0; i < NUM_INPUTS; i++) {
		float in;
		if (!(bss >> in)) {
			std::cout << "Received invalid string from pipe" << std::endl;
			throw std::runtime_error("Received invalid string from pipe");
		}
		inputs.push_back(in);
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

	float output = calculateSingleOutput(layers, inputs);
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
