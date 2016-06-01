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

void loadLocalParameters(std::string parName);

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
	std::cout << "Reading trainset: ";
	if (testUseSampleFile) {
		auto readstart = std::chrono::high_resolution_clock::now();
		size_t numDiscards[2];

		sampleReadTrainSet(trainstring, discardSamples, numDiscards);

		auto readelapsed = std::chrono::high_resolution_clock::now() - readstart;
		long long readtime = std::chrono::duration_cast<std::chrono::microseconds>(readelapsed).count();
		std::cout << readtime / 1000000 << " s" << std::endl;

		std::cout << numDiscards[0] << "/" << numDiscards[1] << " samples discarded" << std::endl;

		numSamples = numDiscards[1] - numDiscards[0];
	}
	else {
		auto readstart = std::chrono::high_resolution_clock::now();

		numSamples = readTrainSet(trainstring);

		auto readelapsed = std::chrono::high_resolution_clock::now() - readstart;
		long long readtime = std::chrono::duration_cast<std::chrono::microseconds>(readelapsed).count();
		std::cout << readtime / 1000000 << " s" << std::endl;
	}

	std::cout << numSamples << " samples read" << std::endl;

	randomizeTrainSet();

	saveExplicitTrainSet(randtrainstring);

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
	}
}
