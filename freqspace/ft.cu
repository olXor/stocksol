#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>

#define LOCAL

std::string trainstring;
std::string freqstring;
std::string pcastring;
std::string PCAWeights;
#ifdef LOCAL
std::string datastring = "rawdata/";
#else
std::string datastring = "../stockproj/rawdata/";
#endif

size_t numFrequencies = 100;
size_t numDims = 200;
size_t numPCAVectors = 20;

size_t minPCAIterations = 100;
size_t maxPCAIterations = 1000;
float PCAEigenErrorThresh = 1e-5;

bool PCASubtractMean = true;
bool PCANormalize = true;

size_t numFreqIntervalSize = 0;
bool freqIncludePhaseInformation = true;

struct FreqComp {
	float real = 0.0f;
	float imag = 0.0f;
	float ampl = 0.0f;
	float phase = 0.0f;
};

void addPosition(std::vector<FreqComp>* freqs, float in, size_t pos);
void normalizeFrequencies(std::vector<FreqComp>* freqs);
void computePCAWeights(std::vector<std::vector<float>> data);
void loadParameters(std::string parName);
std::vector<float> computeMaxEigenVector(std::vector<std::vector<float>> covar);
std::vector<std::vector<float>> loadPCAWeights(std::vector<float>* means, std::vector<float>* PCAMins, std::vector<float>* PCAMaxes);
std::vector<std::vector<float>> createDataMatrix(std::vector<std::vector<FreqComp>> freqs);
void savePCA(std::vector<std::vector<float>> data, std::vector<std::vector<float>> eigenvectors, std::vector<float> correct, std::vector<float> means, std::vector<float> PCAMins, std::vector<float> PCAMaxes);

int main() {
#ifdef LOCAL
	loadParameters("pars.cfg");
#else
	loadParameters("../stockproj/pars.cfg");
#endif

	std::cout << "Would you like to create new PCA weights? (Y/n)" << std::endl;
	char inp;
	bool newPCAWeights = false;
	std::cin >> inp;
	if (inp == 'Y') {
		std::cout << "Are you sure? (Y/n)" << std::endl;
		std::cin >> inp;
		if (inp == 'Y')
			newPCAWeights = true;
	}
	std::stringstream freqoutss;
	freqoutss << datastring << freqstring;
	std::ofstream freqout(freqoutss.str().c_str());

	std::stringstream trainsetss;
	trainsetss << datastring << trainstring;
	std::ifstream trainset(trainsetss.str().c_str());
	std::string line;

	std::cout << "Transforming trainset \"" << trainsetss.str() << "\"" << std::endl;

	std::vector<float> correctoutputs;
	std::vector<std::vector<FreqComp>> freqs;
	size_t numSamples = 0;
	while (getline(trainset, line)) {
		std::stringstream lss(line);
		std::string fname;
		size_t column;
		size_t sstart;
		size_t send;
		float correctoutput;

		lss >> fname;
		lss >> column;
		lss >> sstart;
		lss >> send;
		lss >> correctoutput;


		std::stringstream fullfname;
		fullfname << datastring << fname;
		std::ifstream datafile(fullfname.str().c_str());
		if (!datafile.is_open()) {
			std::stringstream errormsg;
			errormsg << "Couldn't open file " << fullfname.str() << std::endl;
			throw std::runtime_error(errormsg.str().c_str());
		}
		std::string dline;

		std::vector<FreqComp> freqamps;
		freqamps.resize(numFrequencies);

		size_t numPos = 0;
		for (size_t i = 1; i <= send && getline(datafile, dline); i++) {
			if (i < sstart)
				continue;

			std::string dum;
			std::stringstream dliness(dline);
			float in;
			for (size_t j = 0; j < column - 1; j++)
				dliness >> dum;
			dliness >> in;

			addPosition(&freqamps, in, i - sstart);
			numPos++;
			if (numPos == numFreqIntervalSize || (numFreqIntervalSize==0 && i==send)) {
				normalizeFrequencies(&freqamps);
				freqs.push_back(freqamps);
				correctoutputs.push_back(correctoutput);
				freqout << correctoutput << " | ";
				for (size_t i = 0; i < numFrequencies; i++) {
					freqout << freqamps[i].ampl << " ";
					if(freqIncludePhaseInformation)
						freqout << freqamps[i].phase << " ";
				}
				freqout << std::endl;
				numSamples++;
				freqamps.clear();
				freqamps.resize(numFrequencies);
				numPos = 0;
			}
		}
	}
	std::cout << "Fourier transformed " << numSamples << " samples" << std::endl;

	std::vector<std::vector<float>> data = createDataMatrix(freqs);
	if (newPCAWeights) {
		computePCAWeights(data);
	}
	std::vector<float> means;
	std::vector<float> PCAMins;
	std::vector<float> PCAMaxes;
	std::vector<std::vector<float>> pcaWeights = loadPCAWeights(&means, &PCAMins, &PCAMaxes);
	savePCA(data, pcaWeights, correctoutputs, means, PCAMins, PCAMaxes);
#ifdef LOCAL
	system("pause");
#endif
}

void addPosition(std::vector<FreqComp>* freqs, float in, size_t pos) {
	for (size_t k = 0; k < numFrequencies; k++) {
		float angle = -2.0f*M_PI*pos*(k+1) / numFrequencies;
		(*freqs)[k].real += in*cos(angle);
		(*freqs)[k].imag += in*sin(angle);
	}
}

void normalizeFrequencies(std::vector<FreqComp>* freqs) {
	float maxampsq = 0.0f;
	for (size_t i = 0; i < numFrequencies; i++) {
		float real = (*freqs)[i].real;
		float imag = (*freqs)[i].imag;
		float amp = real*real + imag*imag;
		if (amp > maxampsq)
			maxampsq = amp;
		(*freqs)[i].ampl = sqrt(amp);
		if (real > 0 && imag >= 0)
			(*freqs)[i].phase = atan(imag / real);
		else if (real > 0 && imag < 0)
			(*freqs)[i].phase = (float)2.0f * M_PI + atan(imag / real);
		else if (real < 0)
			(*freqs)[i].phase = (float)M_PI - atan(imag / real);
		else if (imag > 0)
			(*freqs)[i].phase = (float)M_PI / 2;
		else
			(*freqs)[i].phase = -(float)M_PI / 2;
	}
	float maxamp = sqrt(maxampsq);
	for (size_t i = 0; i < numFrequencies; i++) {
		(*freqs)[i].real = (*freqs)[i].real / maxamp;
		(*freqs)[i].imag = (*freqs)[i].imag / maxamp;
		(*freqs)[i].ampl = (*freqs)[i].ampl / maxamp;
	}
}

void loadParameters(std::string parName) {
	std::ifstream infile(parName.c_str());
	std::string line;
	while (getline(infile, line)) {
		std::stringstream lss(line);
		std::string var;
		lss >> var;
		if (var == "trainstring")
			lss >> trainstring;
		else if (var == "numFrequencies")
			lss >> numFrequencies;
		else if (var == "numPCAVectors")
			lss >> numPCAVectors;
		else if (var == "PCAWeights")
			lss >> PCAWeights;
		else if (var == "minPCAIterations")
			lss >> minPCAIterations;
		else if (var == "maxPCAIterations")
			lss >> maxPCAIterations;
		else if (var == "PCAEigenErrorThresh")
			lss >> PCAEigenErrorThresh;
		else if (var == "PCASubtractMean")
			lss >> PCASubtractMean;
		else if (var == "numFreqIntervalSize")
			lss >> numFreqIntervalSize;
		else if (var == "freqIncludePhaseInformation")
			lss >> freqIncludePhaseInformation;
		else if (var == "PCANormalize")
			lss >> PCANormalize;
	}

	pcastring = trainstring + "PCA";
	freqstring = trainstring + "freq";

	if (freqIncludePhaseInformation)
		numDims = numFrequencies * 2;
	else
		numDims = numFrequencies;
}

void computePCAWeights(std::vector<std::vector<float>> data) {
	//the PCA weights file also includes the data mean vector
	std::stringstream pcaweightsss;
	pcaweightsss << datastring << PCAWeights;
	std::ofstream pcaweights(pcaweightsss.str().c_str());

	//center data
	for (size_t i = 0; i < numDims; i++) {
		float mean = 0.0f;
		for (size_t j = 0; j < data.size(); j++) {
			mean += data[j][i];
		}
		mean /= data.size();
		for (size_t j = 0; j < data.size(); j++) {
			data[j][i] -= mean;
		}
		pcaweights << mean << " ";
	}
	pcaweights << std::endl;

	std::vector<std::vector<float>> origData = data;	//mean-subtracted

	std::vector<std::vector<float>> covar;
	covar.resize(numDims);
	for (size_t i = 0; i < numDims; i++)
		covar[i].resize(numDims);

	std::vector<std::vector<float>> eigenvectors;
	for (size_t i = 0; i < std::min(numPCAVectors, numDims); i++) {
		for (size_t j = 0; j < numDims; j++) {
			for (size_t k = 0; k < numDims; k++) {
				covar[j][k] = 0.0f;
				for (size_t l = 0; l < data.size(); l++)
					covar[j][k] += data[l][j] * data[l][k];
			}
		}

		std::cout << "Computing eigenvector #" << i + 1;
		std::vector<float> lasteigen = computeMaxEigenVector(covar);
		eigenvectors.push_back(lasteigen);
		std::vector<std::vector<float>> tmp = data;
		for (size_t j = 0; j < data.size(); j++) {
			float proj = 0.0f;
			for (size_t k = 0; k < numDims; k++) {
				proj += data[j][k] * lasteigen[k];
			}
			for (size_t k = 0; k < numDims; k++) {
				tmp[j][k] -= proj*lasteigen[k];
			}
		}
		data = tmp;
	}

	//we compute the projections simply to save the range of each PCA variable; a little inefficient but whatever
	std::vector<float> pcaMins;
	std::vector<float> pcaMaxes;
	pcaMins.resize(eigenvectors.size());
	pcaMaxes.resize(eigenvectors.size());
	for (size_t i = 0; i < eigenvectors.size(); i++) {
		pcaMins[i] = 99999.0f;
		pcaMaxes[i] = -99999.0f;
	}
	for (size_t i = 0; i < origData.size(); i++) {
		for (size_t j = 0; j < eigenvectors.size(); j++) {
			float proj = 0.0f;
			for (size_t k = 0; k < numDims; k++) {
				proj += origData[i][k] * eigenvectors[j][k];
			}
			if (proj < pcaMins[j])
				pcaMins[j] = proj;
			if (proj > pcaMaxes[j])
				pcaMaxes[j] = proj;
		}
	}

	for (size_t i = 0; i < eigenvectors.size(); i++) {
		pcaweights << pcaMins[i] << " ";
	}
	pcaweights << std::endl;

	for (size_t i = 0; i < eigenvectors.size(); i++) {
		pcaweights << pcaMaxes[i] << " ";
	}
	pcaweights << std::endl;

	for (size_t i = 0; i < eigenvectors.size(); i++) {
		for (size_t j = 0; j < numDims; j++) {
			pcaweights << eigenvectors[i][j] << " ";
		}
		pcaweights << std::endl;
	}
}

std::vector<float> computeMaxEigenVector(std::vector<std::vector<float>> covar) {
	std::vector<float> eigen;
	eigen.resize(covar.size());
	std::vector<float> tmp;
	tmp.resize(covar.size());
	float ampl = 0.0f;
	for (size_t i = 0; i < eigen.size(); i++) {
		eigen[i] = (rand() % 11 - 5.0f)/5.0f;
		ampl += eigen[i] * eigen[i];
	}
	ampl = sqrt(ampl);
	for (size_t i = 0; i < eigen.size(); i++) {
		eigen[i] /= ampl;
	}

	ampl = 0.0f;
	float diff;
	for (size_t i = 0; i < maxPCAIterations; i++) {
		for (size_t j = 0; j < tmp.size(); j++) {
			tmp[j] = 0.0f;
			for (size_t k = 0; k < tmp.size(); k++) {
				tmp[j] += covar[j][k] * eigen[k];
			}
			ampl += tmp[j] * tmp[j];
		}
		ampl = sqrt(ampl);
		diff = 0.0f;
		for (size_t j = 0; j < tmp.size(); j++) {
			tmp[j] /= ampl;
			float td = tmp[j] - eigen[j];
			diff += td*td;
		}
		diff = sqrt(diff);
		eigen = tmp;
		if (i > minPCAIterations && diff < PCAEigenErrorThresh) {
			std::cout << " Converged to " << diff << " error after " << i << " iterations" << std::endl;
			return eigen;
		}
	}
	std::cout << " Failed to converge at " << diff << " error after " << maxPCAIterations << " iterations " << std::endl;
	return eigen;
}

std::vector<std::vector<float>> loadPCAWeights(std::vector<float>* means, std::vector<float>* PCAMins, std::vector<float>* PCAMaxes) {
	means->resize(numDims);
	size_t numPCA = std::min(numPCAVectors, numDims);
	PCAMins->resize(numPCA);
	PCAMaxes->resize(numPCA);
	std::stringstream wss;
	wss << datastring << PCAWeights;
	std::ifstream weightin(wss.str().c_str());
	std::string line;

	if (getline(weightin, line)) {
		std::stringstream lss(line);
		for (size_t i = 0; i < numDims; i++) {
			float mean;
			lss >> mean;
			(*means)[i] = mean;
		}
	}

	if (getline(weightin, line)) {
		std::stringstream lss(line);
		for (size_t i = 0; i < numPCA; i++) {
			float pcaMin;
			lss >> pcaMin;
			(*PCAMins)[i] = pcaMin;
		}
	}

	if (getline(weightin, line)) {
		std::stringstream lss(line);
		for (size_t i = 0; i < numPCA; i++) {
			float pcaMax;
			lss >> pcaMax;
			(*PCAMaxes)[i] = pcaMax;
		}
	}

	std::vector<std::vector<float>> weights;
	while (getline(weightin, line)) {
		std::stringstream lss(line);
		std::vector<float> w;
		for (size_t i = 0; i < numDims; i++) {
			float tmp;
			lss >> tmp;
			w.push_back(tmp);
		}
		weights.push_back(w);
	}
	return weights;
}

void savePCA(std::vector<std::vector<float>> data, std::vector<std::vector<float>> eigenvectors, std::vector<float> correct, std::vector<float> means, std::vector<float> PCAMins, std::vector<float> PCAMaxes) {
	std::stringstream pcass;
	pcass << datastring << pcastring;
	std::ofstream pcaout(pcass.str().c_str());

	for (size_t i = 0; i < data.size(); i++) {
		pcaout << correct[i] << " | ";
		if (PCASubtractMean) {
			for (size_t j = 0; j < numDims; j++) {
				data[i][j] -= means[j];
			}
		}
		for (size_t j = 0; j < eigenvectors.size(); j++) {
			float proj = 0.0f;
			for (size_t k = 0; k < numDims; k++) {
				proj += data[i][k] * eigenvectors[j][k];
			}
			if (PCANormalize) {
				if (PCAMaxes[j] - PCAMins[j] != 0)
					proj = 2.0f*(proj - PCAMins[j]) / (PCAMaxes[j] - PCAMins[j]) - 1.0f;
				else
					proj = 0.0f;
			}
			pcaout << proj << " ";
		}
		pcaout << std::endl;
	}
}

std::vector<std::vector<float>> createDataMatrix(std::vector<std::vector<FreqComp>> freqs) {
	std::vector<std::vector<float>> data;
	data.resize(freqs.size());
	for (size_t i = 0; i < data.size(); i++) {
		data[i].resize(numDims);
		for (size_t j = 0; j < numFrequencies; j++) {
			data[i][j] = 2.0f * freqs[i][j].ampl - 1.0f;
			if (freqIncludePhaseInformation)
				data[i][numFrequencies + j] = freqs[i][j].phase / (M_PI) - 1.0f;
		}
	}
	return data;
}
