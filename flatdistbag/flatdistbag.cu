#include <stockrun.cuh>

#ifdef LOCAL
#define datastring "rawdata/"
#define savestring "saveweights/"
#else
#define datastring "../stockproj/rawdata/"
#define savestring "../stockproj/saveweights/"
#endif

std::string flatDistBagSet = "flatdistbagset";
size_t flatDistBagSetSize = 100000;
size_t flatDistBagSetNum = 1;

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

	std::cout << "Reading all samples from trainset: ";

	auto readstart = std::chrono::high_resolution_clock::now();
	size_t totalSamples = readExplicitTrainSet(randtrainstring, 1, 0);

	auto readelapsed = std::chrono::high_resolution_clock::now() - readstart;
	long long readtime = std::chrono::duration_cast<std::chrono::microseconds>(readelapsed).count();
	std::cout << readtime / 1000000 << " s" << std::endl;
	std::cout << totalSamples << " samples loaded" << std::endl;

	std::vector<std::vector<IOPair>> binset = getBinnedTrainset();

	std::cout << "Sample distribution by bin: " << std::endl;
	for (size_t i = 0; i < numBins; i++) {
		std::cout << binMin + i*binWidth << "-" << binMin + (i + 1)*binWidth << ": " << binset[i].size() << std::endl;
	}

	std::cout << "Creating " << flatDistBagSetNum << " flat distribution sets" << std::endl;
	for (size_t n = 0; n < flatDistBagSetNum; n++) {
		std::cout << "Creating flat distribution set #" << n+1 << " of size " << flatDistBagSetSize << ": ";

		std::vector<size_t> binCount;
		binCount.resize(numBins);
		for (size_t i = 0; i < numBins; i++)
			binCount[i] = 0;

		std::stringstream flatsetss;
		flatsetss << datastring << flatDistBagSet << n+1;
		std::ofstream outfile(flatsetss.str().c_str());

		auto createsetstart = std::chrono::high_resolution_clock::now();

		for (size_t i = 0; i < flatDistBagSetSize; i++) {
			size_t randBin;
			do {
				randBin = rand() % numBins;
			} while (binset[randBin].size() == 0);

			size_t randSampleNum = rand() % binset[randBin].size();

			binCount[randBin]++;

			IOPair sample = binset[randBin][randSampleNum];
			outfile << sample.correctoutput*OUTPUT_DIVISOR << " | ";
			for (size_t j = 0; j < sample.inputs.size(); j++) {
				outfile << sample.inputs[j] << " ";
			}
			outfile << std::endl;
		}

		auto createsetelapsed = std::chrono::high_resolution_clock::now() - createsetstart;
		long long createsettime = std::chrono::duration_cast<std::chrono::microseconds>(createsetelapsed).count();
		std::cout << createsettime / 1000000 << " s" << std::endl;

		std::cout << "New set distribution by bin: " << std::endl;
		for (size_t i = 0; i < numBins; i++) {
			std::cout << binMin + i*binWidth << "-" << binMin + (i + 1)*binWidth << ": " << binCount[i] << std::endl;
		}
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
		if (var == "flatDistBagSet")
			lss >> flatDistBagSet;
		else if (var == "flatDistBagSetSize")
			lss >> flatDistBagSetSize;
		else if (var == "flatDistBagSetNum")
			lss >> flatDistBagSetNum;

		numBins = (size_t)((binMax - binMin) / binWidth + 1);
	}
}
