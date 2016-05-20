#include "stockrun.cuh"

#define datastring "../stockproj/rawdata/"
#define savestring "../stockproj/saveweights/"

int main() {
	srand((size_t)time(NULL));
	setStrings(datastring, savestring);

	size_t numSamples = readTrainSet(trainstring);
	std::cout << numSamples << " samples read" << std::endl;

	randomizeTrainSet();

	saveExplicitTrainSet(randtrainstring);
}