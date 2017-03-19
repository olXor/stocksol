#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <ctime>
#include <random>
#include <algorithm>

#define DERIV_SMEAR_WIDTH 6
#define DERIV_SMEAR (2*DERIV_SMEAR_WIDTH + 1)
#define DERIV2_SMEAR_WIDTH 3
#define DERIV2_SMEAR (2*DERIV_SMEAR_WIDTH + 1)
#define BASELINE_AVERAGE_WINDOW_WIDTH 40	//on each side
#define BASELINE_AVERAGE_WINDOW (2*BASELINE_AVERAGE_WINDOW_WIDTH + 1)

#define START_D2_THRESH 0.008
#define START_ABS_THRESH -0.7

#define MIN_PULSE_SIZE 50
#define MAX_PULSE_SIZE 100

#define END_MIN_SEARCH 20

#define NUM_INPUTS 136

std::string datastring = "rawdata/";

bool averageIntervals = false;
std::vector<std::vector<float>> intervalPulses;

size_t getPulses(std::vector<float> inputs, std::ofstream* pulsefile, float correctoutput);
void savePulse(float* inputs, std::ofstream* pulsefile, float correctoutput, size_t start, size_t end);
float getSpline(float left, float right, float leftderiv, float rightderiv, float place);
void saveCombinedPulse(std::ofstream* pulsefile, float correctoutput);

int main() {
	srand((size_t)time(NULL));

	std::string learnsetname;
	std::string pulsename;

	std::cout << "Enter interval summary file name: ";
	std::cin >> learnsetname;
	std::cout << std::endl;

	std::cout << "Enter pulse file name: ";
	std::cin >> pulsename;
	std::cout << std::endl;

	std::cout << "Average pulses in the same interval? ";
	std::cin >> averageIntervals;
	std::cout << std::endl;

	std::stringstream pulsess;
	pulsess << datastring << pulsename;
	std::ofstream pulsefile(pulsess.str());

	std::stringstream learnss;
	learnss << datastring << learnsetname;
	std::ifstream learnset(learnss.str());
	std::string line;
	size_t numIntervals = 0;
	size_t numPulses = 0;
	size_t numEmptyIntervals = 0;
	while (getline(learnset, line)) {
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
		std::cout << fullfname.str() << std::endl;
		std::ifstream datafile(fullfname.str().c_str());
		if (!datafile.is_open()) {
			std::stringstream errormsg;
			errormsg << "Couldn't open file " << fullfname.str() << std::endl;
			throw std::runtime_error(errormsg.str().c_str());
		}
		std::string dline;

		std::vector<float> inputs;
		for (size_t i = 1; i < sstart; i++) {
			if (!datafile.ignore(10000000, datafile.widen('\n'))) {
				std::cout << "Error skipping past initial lines in data file" << std::endl;
			}
		}
		for (size_t i = sstart; i <= send && getline(datafile, dline); i++) {
			std::string dum;
			std::stringstream dliness(dline);
			float in;
			for (size_t j = 0; j < column - 1; j++)
				dliness >> dum;
			dliness >> in;
			inputs.push_back(in);
		}

		size_t intPulses = getPulses(inputs, &pulsefile, correctoutput);
		numIntervals++;
		if (intPulses == 0)
			numEmptyIntervals++;
		numPulses += intPulses;
	}

	std::cout << numPulses << " pulses in " << numIntervals << " intervals (" << 1.0f*numPulses / numIntervals << ") with " << numEmptyIntervals << " intervals with no pulses found." << std::endl;

	system("pause");
}

size_t getPulses(std::vector<float> inputs, std::ofstream* pulsefile, float correctoutput) {
	if (inputs.size() < BASELINE_AVERAGE_WINDOW) {
		std::cout << "Intervals need to be at least size " << BASELINE_AVERAGE_WINDOW << std::endl;
		return 0;
	}

	size_t numWindows = inputs.size() - BASELINE_AVERAGE_WINDOW + 1;

	std::vector<float> averages;
	std::vector<float> stdevs;

	float movingaverage = 0;
	for (size_t i = 0; i < BASELINE_AVERAGE_WINDOW; i++) {
		movingaverage += inputs[i] / (BASELINE_AVERAGE_WINDOW);
	}

	for (size_t i = 0; i < numWindows-1; i++) {
		averages.push_back(movingaverage);
		movingaverage += inputs[i + BASELINE_AVERAGE_WINDOW] / (BASELINE_AVERAGE_WINDOW);
		movingaverage -= inputs[i] / (BASELINE_AVERAGE_WINDOW);
	}
	averages.push_back(movingaverage);

	float movingsqerr = 0;
	for (size_t i = 0; i < BASELINE_AVERAGE_WINDOW; i++) {
		float unsquare = inputs[i] - averages[std::min(i-BASELINE_AVERAGE_WINDOW_WIDTH,numWindows-1)];
		movingsqerr += unsquare*unsquare / (BASELINE_AVERAGE_WINDOW);
	}

	for (size_t i = 0; i < numWindows - 1; i++) {
		stdevs.push_back(sqrt(movingsqerr));
		float unsquare = inputs[i+BASELINE_AVERAGE_WINDOW] - averages[std::min(i+BASELINE_AVERAGE_WINDOW_WIDTH,numWindows-1)];
		movingsqerr += unsquare*unsquare / (BASELINE_AVERAGE_WINDOW);
		unsquare = inputs[i] - averages[std::min(i-BASELINE_AVERAGE_WINDOW_WIDTH,numWindows-1)];
		movingsqerr -= unsquare*unsquare / (BASELINE_AVERAGE_WINDOW);
	}
	stdevs.push_back(sqrt(movingsqerr));

	std::vector<float> derivs;
	float smallmovavg = 0;
	for (size_t i = 0; i < DERIV_SMEAR; i++) {
		smallmovavg += inputs[i] / (DERIV_SMEAR);
	}

	for (size_t i = 0; i < inputs.size()-DERIV_SMEAR; i++) {
		derivs.push_back((inputs[i + DERIV_SMEAR]-inputs[i])/DERIV_SMEAR);
	}

	size_t numPulses = 0;
	size_t lastTrigger = 0;
	for (size_t i = 0; i < numWindows; i++) {
		size_t dIndex = i + BASELINE_AVERAGE_WINDOW_WIDTH - DERIV_SMEAR_WIDTH;
		size_t iIndex = i + BASELINE_AVERAGE_WINDOW_WIDTH;
		float d2crit = (derivs[dIndex + DERIV2_SMEAR_WIDTH] - derivs[dIndex - DERIV2_SMEAR_WIDTH]) / (DERIV2_SMEAR-1) / stdevs[i];
		float absCrit = (inputs[iIndex] - averages[i]) / stdevs[i];

		if (d2crit >= START_D2_THRESH && absCrit <= START_ABS_THRESH) {
			if (lastTrigger != 0 && i - lastTrigger >= MIN_PULSE_SIZE && i - lastTrigger <= MAX_PULSE_SIZE) {
				savePulse(&inputs[BASELINE_AVERAGE_WINDOW_WIDTH], pulsefile, correctoutput, lastTrigger, i);
				numPulses++;
			}
			if (lastTrigger == 0 || i - lastTrigger >= MIN_PULSE_SIZE)
				lastTrigger = i;
		}
	}

	if (averageIntervals)
		saveCombinedPulse(pulsefile, correctoutput);

	/*
	std::ofstream outtest("pulsetest");
	size_t lastTrigger2 = 0;
	for (size_t i = 0; i < numWindows; i++) {
		size_t dIndex = i + BASELINE_AVERAGE_WINDOW_WIDTH - DERIV_SMEAR_WIDTH;
		size_t iIndex = i + BASELINE_AVERAGE_WINDOW_WIDTH;
		std::cout << "(" << inputs[iIndex] << ", " << (derivs[dIndex + 1] - derivs[dIndex - 1]) / stdevs[i] << ", " << inputs[iIndex] - averages[i] << "/" << stdevs[i] << ") ";
		float d2crit = (derivs[dIndex + DERIV2_SMEAR_WIDTH] - derivs[dIndex - DERIV2_SMEAR_WIDTH]) / (DERIV2_SMEAR-1) / stdevs[i];
		float absCrit = (inputs[iIndex] - averages[i]) / stdevs[i];

		outtest << i << " " << inputs[iIndex] << " " << 20.0f*d2crit << " " << absCrit << " ";
		if (d2crit >= START_D2_THRESH && absCrit <= START_ABS_THRESH) {
			outtest << 1.0f << " ";
			if (lastTrigger2 == 0 || (i - lastTrigger2 >= MIN_PULSE_SIZE && i - lastTrigger2 <= MAX_PULSE_SIZE))
				outtest << 1.0f;
			else if (lastTrigger2 != 0 && i - lastTrigger2 > MAX_PULSE_SIZE)
				outtest << -1.0f;
			else outtest << 0.0f;
			if (lastTrigger2 == 0 || i - lastTrigger2 >= MIN_PULSE_SIZE)
				lastTrigger2 = i;
		}
		else outtest << 0.0f << " " << 0.0f;
		outtest << std::endl;
	}
	std::cout << std::endl;
	*/
	return numPulses;
}

void savePulse(float* inputs, std::ofstream* pulsefile, float correctoutput, size_t start, size_t end) {
	float startMin = 999999;
	size_t startMinIndex = 0;
	float endMin = 999999;
	size_t endMinIndex = 0;

	for (size_t i = start; i < start + END_MIN_SEARCH && i < end; i++) {
		if (inputs[i] < startMin) {
			startMin = inputs[i];
			startMinIndex = i;
		}
	}
	for (size_t i = end - 1; i >= end - END_MIN_SEARCH && i >= start && i < end; i--) {
		if (inputs[i] < endMin) {
			endMin = inputs[i];
			endMinIndex = i;
		}
	}

	size_t length = end - start;
	std::vector<float> orig(length);
	std::vector<float> origderivs(length);
	float slope = (endMin - startMin) / (1.0f*(endMinIndex - startMinIndex));
	for (size_t i = 0; i < length; i++) {
		orig[i] = inputs[start + i] - slope*i;
	}

	//second round of baseline subtraction for weird cases where the minimums shift due to very high baselines.
	startMin = 999999;
	startMinIndex = 0;
	endMin = 999999;
	endMinIndex = 0;
	for (size_t i = 0; i < END_MIN_SEARCH && i < length; i++) {
		if (orig[i] < startMin) {
			startMin = orig[i];
			startMinIndex = i;
		}
	}
	for (size_t i = length - 1; i >= length - END_MIN_SEARCH && i < length; i--) {
		if (orig[i] < endMin) {
			endMin = orig[i];
			endMinIndex = i;
		}
	}
	slope = (endMin - startMin) / (1.0f*(endMinIndex - startMinIndex));
	for (size_t i = 0; i < length; i++) {
		orig[i] = orig[i] - slope*i;
	}

	for (size_t i = 0; i < length; i++) {
		if (i == 0)
			origderivs[i] = orig[i + 1] - orig[i];
		else if (i == length - 1)
			origderivs[i] = orig[i] - orig[i - 1];
		else
			origderivs[i] = (orig[i + 1] - orig[i - 1]) / 2;
	}

	std::vector<float> resampled(NUM_INPUTS);
	for (size_t i = 0; i < NUM_INPUTS; i++) {
		float origpoint = 1.0f*(length-1)*i / (NUM_INPUTS-1);
		size_t origindex = (size_t)origpoint;
		if (origindex >= length - 1)
			origindex = length - 2;
		float remainder = origpoint - (float)origindex;

		resampled[i] = getSpline(orig[origindex], orig[origindex + 1], origderivs[origindex], origderivs[origindex + 1], remainder);
	}

	float max = -999999;
	float min = 999999;
	for (size_t i = 0; i < NUM_INPUTS; i++) {
		if (resampled[i] > max)
			max = resampled[i];
		if (resampled[i] < min)
			min = resampled[i];
	}
	for (size_t i = 0; i < NUM_INPUTS; i++) {
		resampled[i] = 2*(resampled[i] - min) / (max - min) - 1.0f;
	}

	/*
	//(*pulsefile) << correctoutput << " | ";
	(*pulsefile) << "\"" << start << "\"" << std::endl;
	for (size_t i = 0; i < NUM_INPUTS; i++) {
		(*pulsefile) << resampled[i] << std::endl;
	}
	(*pulsefile) << std::endl << std::endl;
	*/
	if (!averageIntervals) {
		(*pulsefile) << correctoutput << " | ";
		for (size_t i = 0; i < NUM_INPUTS; i++) {
			(*pulsefile) << resampled[i] << " ";
		}
		(*pulsefile) << std::endl;
	}
	else
		intervalPulses.push_back(resampled);
}

float getSpline(float left, float right, float leftderiv, float rightderiv, float place) {
	float a = leftderiv - (right - left);
	float b = -rightderiv + (right - left);
	return (1 - place)*left + place*right + place*(1 - place)*(a*(1 - place) + b*place);
}

void saveCombinedPulse(std::ofstream* pulsefile, float correctoutput) {
	if (intervalPulses.size() == 0)
		return;

	std::vector<float> average(NUM_INPUTS);
	for (size_t j = 0; j < NUM_INPUTS; j++) {
		for (size_t i = 0; i < intervalPulses.size(); i++) {
			average[j] += intervalPulses[i][j];
		}
		average[j] /= intervalPulses.size();
	}

	float max = -999999;
	float min = 999999;
	for (size_t i = 0; i < NUM_INPUTS; i++) {
		if (average[i] > max)
			max = average[i];
		if (average[i] < min)
			min = average[i];
	}
	for (size_t i = 0; i < NUM_INPUTS; i++) {
		average[i] = 2*(average[i] - min) / (max - min) - 1.0f;
	}

	(*pulsefile) << correctoutput << " | ";
	for (size_t i = 0; i < NUM_INPUTS; i++) {
		(*pulsefile) << average[i] << " ";
	}
	(*pulsefile) << std::endl;

	intervalPulses.clear();
}