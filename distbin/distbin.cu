#include <stockrun.cuh>
#include "Shlwapi.h"

#ifdef LOCAL
#define datastring "rawdata/"
#define savestring "saveweights/"
#else
#define datastring "../stockproj/rawdata/"
#define savestring "../stockproj/saveweights/"
#endif

void readData();
void quicksort(std::vector<float>* A, size_t lo, size_t hi);

int main() {
	srand((size_t)time(NULL));
	setStrings(datastring, savestring);
#ifdef LOCAL
	loadParameters("pars.cfg");
#else
	loadParameters("../stockproj/pars.cfg");
#endif

	size_t binNum = 5;
	std::cout << "Number of bins? ";
	std::cin >> binNum;
	std::cout << std::endl;

	std::string binfile;
	std::cout << "Bin edge save file? ";
	std::cin >> binfile;
	std::cout << std::endl;

	readData();

	std::cout << "Sorting: ";
	markTime();
	std::vector<float> values;
	for (size_t i = 0; i < getTrainSet()->size(); i++) {
		values.push_back((*getTrainSet())[i].correctoutput);
	}
	quicksort(&values, 0, values.size()-1);
	std::cout << " Done. (" << getTimeSinceMark() << " s)" << std::endl;

	std::vector<float> binEdges(binNum-1);
	for (size_t i = 0; i < binNum - 1; i++) {
		binEdges[i] = values[(i + 1)*values.size() / binNum];
	}

	std::cout << "Bin Edges: ";
	for (size_t i = 0; i < binNum - 1; i++)
		std::cout << binEdges[i] << " ";
	std::cout << std::endl;

	std::stringstream outss;
	outss << datastring << binfile;
	std::ofstream binout(outss.str());
	for (size_t i = 0; i < binNum - 1; i++)
		binout << binEdges[i] << " ";
	binout << std::endl;

	system("pause");
}

void readData() {
	std::cout << "Reading all samples from " << randtrainstring << ": ";
	markTime();
	size_t totalSamples = readExplicitTrainSet(randtrainstring, 1, 0);
	std::cout << " Done. (" << getTimeSinceMark() << " s)" << std::endl;
	std::cout << totalSamples << " samples loaded." << std::endl;
}

float median(float a, float b, float c) {
	return max(min(a, b), min(max(a, b), c));
}

void swap(std::vector<float>* A, size_t i, size_t j) {
	float tmp = (*A)[i];
	(*A)[i] = (*A)[j];
	(*A)[j] = tmp;
}

void quicksort(std::vector<float>* A, size_t lo, size_t hi) {
	if (lo >= hi)
		return;
	float pivot = median((*A)[lo], (*A)[hi], (*A)[(lo + hi) / 2]);
	size_t i = lo-1;
	size_t j = hi+1;
	while (true) {
		do {
			i++;
		} while (i <= hi && (*A)[i] < pivot);
		do{ 
			j--;
		} while (j >= lo && (*A)[j] > pivot);
		if (i >= j)
			break;
		swap(A, i, j);
	}
	quicksort(A, lo, j);
	quicksort(A, j+1, hi);
}