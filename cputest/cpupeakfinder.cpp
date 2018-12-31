#include "cpupeakfinder.h"

#define MATH_PI 3.14159265358979323846

#define MIN_FIRST_HARMONIC 4
#define MAX_FIRST_HARMONIC 20
#define FIRST_HARMONIC_MIN_HEIGHT 0.50
#define HARMONIC_WIDTH 2
#define HARMONIC_FALLOFF 0.9

#define DEFAULT_FIRST_HARMONIC 5

#define PEAK_FIND_DERIV_SMOOTHING_RANGE 15

#define HARMONIC_PERIOD_SEARCH_START 0.05f
#define HARMONIC_PERIOD_SHORT_SEARCH_END 0.5f
#define HARMONIC_PERIOD_LONG_OVERRIDE_BEGIN 0.8f
#define HARMONIC_PERIOD_LONG_SEARCH_END 1.3f
#define HARMONIC_PERIOD_SHORT_LONG_RATIO_THRESH 0.80
#define PROHIBIT_TWO_SHORTS_IN_ROW 1

#define ADJUST_CLOSE_VALLEY_THRESHOLD 1.5f

#define STARTING_PEAK_THRESHOLD 0.5
#define EDGE_EXCLUSION 0//(1.1*PEAK_FIND_DERIV_SMOOTHING_RANGE)

//#define HARMONIC_DERIV2
#define HARMONIC_BOTH

std::vector<float> peakFindDeriv1;
std::vector<float> peakFindDeriv2;

#define FOURIER_PATCH_WIDTH_POW 9
#define FOURIER_PATCH_WIDTH (pow(2,FOURIER_PATCH_WIDTH_POW))
std::vector<float> fourierRe;
std::vector<float> fourierIm;
std::vector<float> fourierAvg;

#define BAND_PASS_HIGH 200
#define BAND_PASS_LOW 253

/**
* C++ implementation of FFT
*
* Source: http://paulbourke.net/miscellaneous/dft/
*/

/*
This computes an in-place complex-to-complex FFT
x and y are the real and imaginary arrays of 2^m points.
dir =  1 gives forward transform
dir = -1 gives reverse transform
*/
inline void FFT(short int dir, long m, float *x, float *y)
{
	long n, i, i1, j, k, i2, l, l1, l2;
	float c1, c2, tx, ty, t1, t2, u1, u2, z;

	/* Calculate the number of points */
	n = 1;
	for (i = 0; i<m; i++)
		n *= 2;

	/* Do the bit reversal */
	i2 = n >> 1;
	j = 0;
	for (i = 0; i<n - 1; i++) {
		if (i < j) {
			tx = x[i];
			ty = y[i];
			x[i] = x[j];
			y[i] = y[j];
			x[j] = tx;
			y[j] = ty;
		}
		k = i2;
		while (k <= j) {
			j -= k;
			k >>= 1;
		}
		j += k;
	}

	/* Compute the FFT */
	c1 = -1.0;
	c2 = 0.0;
	l2 = 1;
	for (l = 0; l<m; l++) {
		l1 = l2;
		l2 <<= 1;
		u1 = 1.0;
		u2 = 0.0;
		for (j = 0; j<l1; j++) {
			for (i = j; i<n; i += l2) {
				i1 = i + l1;
				t1 = u1 * x[i1] - u2 * y[i1];
				t2 = u1 * y[i1] + u2 * x[i1];
				x[i1] = x[i] - t1;
				y[i1] = y[i] - t2;
				x[i] += t1;
				y[i] += t2;
			}
			z = u1 * c1 - u2 * c2;
			u2 = u1 * c2 + u2 * c1;
			u1 = z;
		}
		c2 = sqrt((1.0 - c1) / 2.0);
		if (dir == 1)
			c2 = -c2;
		c1 = sqrt((1.0 + c1) / 2.0);
	}

	/* Scaling for forward transform */
	if (dir == 1) {
		for (i = 0; i<n; i++) {
			x[i] /= n;
			y[i] /= n;
		}
	}
}

size_t findFirstHarmonic(std::vector<float>* freq) {
	float maxAmp = 0;
	for (size_t i = MIN_FIRST_HARMONIC; i < MAX_FIRST_HARMONIC && i < freq->size(); i++) {
		maxAmp = std::max(maxAmp, (*freq)[i]);
	}

	for (size_t i = MIN_FIRST_HARMONIC; i < MAX_FIRST_HARMONIC && i < freq->size(); i++) {
		if ((*freq)[i] >= FIRST_HARMONIC_MIN_HEIGHT*maxAmp) {
			float backMin = 99999;
			float forwardMin = 99999;
			float backMax = -99999;
			float forwardMax = -99999;
			for (size_t j = (i > HARMONIC_WIDTH ? i - HARMONIC_WIDTH : 0); j < i; j++) {
				backMin = std::min(backMin, (*freq)[j]);
				backMax = std::max(backMax, (*freq)[j]);
			}
			for (size_t j = i + 1; j < i + HARMONIC_WIDTH + 1 && j < freq->size(); j++) {
				forwardMin = std::min(forwardMin, (*freq)[j]);
				forwardMax = std::max(forwardMax, (*freq)[j]);
			}
			if (backMin < HARMONIC_FALLOFF*(*freq)[i] && forwardMin < HARMONIC_FALLOFF*(*freq)[i] && backMax < (*freq)[i] && forwardMax < (*freq)[i])
				return i;
		}
	}

	return 0;
}

//peakSaveFactor is the number to store in the peaks array; we can set it to different values on the forward and backwards passes to tell them apart
float searchForPeaksAndValleys(std::vector<float>* deriv, std::vector<float>* peaks, size_t harmonicPeriod, int direction, float peakSaveFactor) {
	bool searchingForPeak = false;

	float begAbsMax = 0;
	float begMin = 99999;
	float begMax = -99999;
	for (size_t i = EDGE_EXCLUSION; i < harmonicPeriod; i++) {
		size_t pos = (direction > 0 ? i : deriv->size() - i - 1);
		begAbsMax = std::max(abs((*deriv)[pos]), begAbsMax);
		begMax = std::max((*deriv)[pos], begMax);
		begMin = std::min((*deriv)[pos], begMin);
	}

	size_t curIndex = EDGE_EXCLUSION;
	size_t lastCritPoint = EDGE_EXCLUSION;
	for (size_t i = EDGE_EXCLUSION; i < harmonicPeriod; i++) {
		size_t pos = (direction > 0 ? i : deriv->size() - i - 1);
		if (abs((*deriv)[pos]) > STARTING_PEAK_THRESHOLD*begAbsMax) {
			float localMax = abs((*deriv)[pos]);
			float originalHit = (*deriv)[pos];
			float maxLoc = i;
			pos += direction;
			i++;
			while ((*deriv)[pos] * originalHit > 0) {
				if (abs((*deriv)[pos]) > localMax) {
					localMax = abs((*deriv)[pos]);
					maxLoc = i;
				}
				pos += direction;
				i++;
			}
			curIndex = maxLoc + HARMONIC_PERIOD_SEARCH_START*harmonicPeriod;
			lastCritPoint = maxLoc;
			size_t maxPos = (direction > 0 ? maxLoc : deriv->size() - maxLoc - 1);
			(*peaks)[maxPos] += ((*deriv)[maxPos] > 0 ? -peakSaveFactor : peakSaveFactor);
			searchingForPeak = ((*deriv)[maxPos] > 0 ? true : false);
			break;
		}
	}

	float begFauxCritVal = (searchingForPeak ? begMax : begMin);

	size_t currentBestIndex = 0;
	float currentBest = 0;
	size_t currentShortBestIndex = 0;
	float currentShortBest = 0;
	float periodMean = 0;
	float periodStdev = 0;
	size_t lastPeak = 0;
	size_t lastValley = 0;
	size_t countPeriods = 0;
	bool onLastInterval = false;
	bool foundCritPoint = false;
	bool lastIntervalWasShort = false;
	while (curIndex < (*deriv).size()) {
		size_t curPos = (direction > 0 ? curIndex : deriv->size() - curIndex - 1);
		//peaks are valleys in second derivative
		if ((searchingForPeak && (*deriv)[curPos] < currentBest) || (!searchingForPeak && (*deriv)[curPos] > currentBest)) {
			currentBest = (*deriv)[curPos];
			currentBestIndex = curIndex;
		}
		if (curIndex < lastCritPoint + HARMONIC_PERIOD_SHORT_SEARCH_END*harmonicPeriod && ((searchingForPeak && (*deriv)[curPos] < currentShortBest) || (!searchingForPeak && (*deriv)[curPos] > currentShortBest))) {
			currentShortBest = (*deriv)[curPos];
			currentShortBestIndex = curIndex;
		}

		curIndex++;
		if (curIndex > std::min((size_t)(lastCritPoint + HARMONIC_PERIOD_LONG_SEARCH_END*harmonicPeriod), (size_t)((*deriv).size() - 1))) {
			size_t lastCritPos = (direction > 0 ? lastCritPoint : deriv->size() - lastCritPoint - 1);
			float lastCritVal;
			if (lastCritPoint == 0)
				lastCritVal = begFauxCritVal;
			else
				lastCritVal = (*deriv)[lastCritPos];
			if (currentShortBest != 0 && abs(currentShortBest - lastCritVal) > HARMONIC_PERIOD_SHORT_LONG_RATIO_THRESH*abs(currentBest - lastCritVal) && currentBestIndex > lastCritPoint + HARMONIC_PERIOD_LONG_OVERRIDE_BEGIN*harmonicPeriod && !(PROHIBIT_TWO_SHORTS_IN_ROW && lastIntervalWasShort)) {
				currentBest = currentShortBest;
				currentBestIndex = currentShortBestIndex;
				lastIntervalWasShort = true;
			}
			else
				lastIntervalWasShort = false;

			if (lastCritPoint != currentBestIndex)
				foundCritPoint = true;
			lastCritPoint = currentBestIndex;
			size_t currentBestPos = (direction > 0 ? currentBestIndex : deriv->size() - currentBestIndex - 1);
			if (currentBest != 0) {
				//if (currentBestIndex > (*deriv).size() / 2)
				(*peaks)[currentBestPos] += (searchingForPeak ? peakSaveFactor : -peakSaveFactor);
				float unsup = 0;
				if (searchingForPeak) {
					if (lastPeak > 0)
						unsup = currentBestIndex - lastPeak;
					lastPeak = currentBestIndex;
				}
				else {
					if (lastValley > 0)
						unsup = currentBestIndex - lastValley;
					lastValley = currentBestIndex;
				}

				if (unsup > 0) {
					periodMean += unsup;
					periodStdev += unsup*unsup;
					countPeriods++;
				}
			}
			else
				searchingForPeak = !searchingForPeak;
			currentBest = 0;
			currentShortBest = 0;
			if (lastCritPoint + harmonicPeriod < deriv->size() && foundCritPoint) {
				curIndex = lastCritPoint + HARMONIC_PERIOD_SEARCH_START*harmonicPeriod;
				searchingForPeak = !searchingForPeak;
				foundCritPoint = false;
			}
			else if (!onLastInterval && foundCritPoint) {
				curIndex = lastCritPoint + HARMONIC_PERIOD_SEARCH_START*harmonicPeriod;
				searchingForPeak = !searchingForPeak;
				onLastInterval = true;
				foundCritPoint = false;
			}
			else
				break;
		}
	}

	periodMean /= countPeriods;
	periodStdev /= countPeriods;
	periodStdev -= periodMean*periodMean;
	periodStdev = sqrt(periodStdev);
	return periodMean;
}

void adjustValleys(std::vector<float>* deriv, std::vector<float>* peaks, size_t harmonicPeriod) {
	std::vector<size_t> peakLocs;
	std::vector<size_t> valleyLocs;

	size_t lastPeak = peaks->size();
	size_t lastValley = peaks->size();
	float peakToValleyDist = 0;
	size_t numPeakToValleys = 0;
	float valleyToPeakDist = 0;
	size_t numValleyToPeaks = 0;
	float averagePeakHeight = 0;
	float averageValleyHeight = 0;
	for (size_t i = 0; i < peaks->size(); i++) {
		if ((*peaks)[i] == 1) {
			peakLocs.push_back(i);
			if (lastValley < peaks->size()) {
				valleyToPeakDist += i - lastValley;
				numValleyToPeaks++;
			}
			averagePeakHeight += abs((*deriv)[i]);
			lastPeak = i;
		}
		else if ((*peaks)[i] == -1) {
			valleyLocs.push_back(i);
			if (lastPeak < peaks->size()) {
				peakToValleyDist += i - lastPeak;
				numPeakToValleys++;
			}
			averageValleyHeight += abs((*deriv)[i]);
			lastValley = i;
		}
	}
	peakToValleyDist /= numPeakToValleys;
	valleyToPeakDist /= numValleyToPeaks;
	averagePeakHeight /= peakLocs.size();
	averageValleyHeight /= valleyLocs.size();

	//bool waveInverted = (peakToValleyDist < valleyToPeakDist ? true : false);
	bool waveInverted = (averageValleyHeight > averagePeakHeight ? true : false);	//can often be wrong if there is only one clear peak and valley per pulse, but that should be taken care of by the close/far adjustment (inversion by horizontal distance is better in those cases)

	std::vector<size_t>* primaryLocs = (!waveInverted ? &peakLocs : &valleyLocs);
	if (primaryLocs->size() == 0)
		return;

	std::vector<size_t> closeOptima;
	std::vector<size_t> farOptima;
	float averageCloseOptimum = 0;	//actually sum of optima
	float averageFarOptimum = 0;
	for (size_t i = 0; i < primaryLocs->size() - 1; i++) {
		float farOptimum = (!waveInverted ? -9999 : 9999);
		size_t farOptimumLoc = (*primaryLocs)[i] + 1;
		float closeOptimum = (!waveInverted ? -9999 : 9999);
		size_t closeOptimumLoc = (*primaryLocs)[i] + 1;

		for (size_t j = (*primaryLocs)[i] + 1; j < (*primaryLocs)[i + 1]; j++) {
			if ((!waveInverted ? (*deriv)[j] > closeOptimum : (*deriv)[j] < closeOptimum) && j - (*primaryLocs)[i] < ((*primaryLocs)[i + 1] - (*primaryLocs)[i]) / 2) {
				closeOptimum = (*deriv)[j];
				closeOptimumLoc = j;
			}
			if ((!waveInverted ? (*deriv)[j] > farOptimum : (*deriv)[j] < farOptimum) && j - (*primaryLocs)[i] >= ((*primaryLocs)[i + 1] - (*primaryLocs)[i]) / 2) {
				farOptimum = (*deriv)[j];
				farOptimumLoc = j;
			}
			if ((*peaks)[j] == (!waveInverted ? -1 : 1)) {
				(*peaks)[j] = 0;
			}
		}
		closeOptima.push_back(closeOptimumLoc);
		farOptima.push_back(farOptimumLoc);
		averageCloseOptimum += abs(closeOptimum);
		averageFarOptimum += abs(farOptimum);
	}

	std::vector<size_t>* optima = (averageCloseOptimum > ADJUST_CLOSE_VALLEY_THRESHOLD*averageFarOptimum ? &closeOptima : &farOptima);
	for (size_t i = 0; i < optima->size(); i++) {
		(*peaks)[(*optima)[i]] = (!waveInverted ? -1 : 1);
	}
}

void findPeaksAndValleys(std::vector<float>* waveform, std::vector<float>* peaks) {
	peaks->clear();
	peaks->resize(waveform->size());

	computeDerivative(waveform, &peakFindDeriv1, PEAK_FIND_DERIV_SMOOTHING_RANGE);
	computeDerivative(&peakFindDeriv1, &peakFindDeriv2, PEAK_FIND_DERIV_SMOOTHING_RANGE);

	fourierRe.clear();
	fourierRe.resize(FOURIER_PATCH_WIDTH);
	fourierIm.clear();
	fourierIm.resize(FOURIER_PATCH_WIDTH);
	fourierAvg.clear();
	fourierAvg.resize(FOURIER_PATCH_WIDTH);

#if defined(HARMONIC_DERIV2) || defined(HARMONIC_BOTH)
	interpolate(&peakFindDeriv2, &fourierAvg, FOURIER_PATCH_WIDTH);
#else
	interpolate(&waveform, &fourierAvg, FOURIER_PATCH_WIDTH);
#endif

	applyBandPassFilter(&fourierAvg, &fourierRe);

	FFT(1, FOURIER_PATCH_WIDTH_POW, &fourierRe[0], &fourierIm[0]);

	for (size_t i = 0; i < fourierRe.size(); i++) {
		fourierRe[i] = sqrt(fourierRe[i] * fourierRe[i] + fourierIm[i] * fourierIm[i]);
	}

	interpolate(&fourierRe, &fourierAvg, FOURIER_PATCH_WIDTH*FOURIER_PATCH_WIDTH / peakFindDeriv2.size());

	size_t peakFindFirstHarmonic = findFirstHarmonic(&fourierAvg);

#if defined(HARMONIC_BOTH)
	interpolate(waveform, &fourierAvg, FOURIER_PATCH_WIDTH);

	applyBandPassFilter(&fourierAvg, &fourierRe);

	FFT(1, FOURIER_PATCH_WIDTH_POW, &fourierRe[0], &fourierIm[0]);

	for (size_t i = 0; i < fourierRe.size(); i++) {
		fourierRe[i] = sqrt(fourierRe[i] * fourierRe[i] + fourierIm[i] * fourierIm[i]);
	}

	interpolate(&fourierRe, &fourierAvg, FOURIER_PATCH_WIDTH*FOURIER_PATCH_WIDTH / waveform->size());

	size_t peakFindFirstHarmonicWaveform = findFirstHarmonic(&fourierAvg);
	if (abs((int)peakFindFirstHarmonicWaveform - (int)DEFAULT_FIRST_HARMONIC) < abs((int)peakFindFirstHarmonic - (int)DEFAULT_FIRST_HARMONIC))
		peakFindFirstHarmonic = peakFindFirstHarmonicWaveform;
#endif

	if (peakFindFirstHarmonic == 0)
		peakFindFirstHarmonic = DEFAULT_FIRST_HARMONIC;

	size_t harmonicPeriod = waveform->size() / peakFindFirstHarmonic;

	float forwardMean = searchForPeaksAndValleys(&peakFindDeriv2, peaks, harmonicPeriod, 1, 1.0f);
	float backwardMean = searchForPeaksAndValleys(&peakFindDeriv2, peaks, harmonicPeriod, -1, 0.1f);

	if (fabs(forwardMean - harmonicPeriod) < fabs(backwardMean - harmonicPeriod)) {
		for (size_t i = 0; i < peaks->size(); i++) {
			if ((*peaks)[i] < -0.5f)
				(*peaks)[i] = -1.0f;
			else if ((*peaks)[i] > 0.5f)
				(*peaks)[i] = 1.0f;
			else
				(*peaks)[i] = 0.0f;
		}
	}
	else {
		for (size_t i = 0; i < peaks->size(); i++) {
			if ((*peaks)[i] < -0.5f)
				(*peaks)[i] += 1.0f;
			else if ((*peaks)[i] > 0.5f)
				(*peaks)[i] -= 1.0f;
			(*peaks)[i] *= 10.0f;
		}
	}

	adjustValleys(&peakFindDeriv2, peaks, harmonicPeriod);
}

void applyBandPassFilter(std::vector<float>* waveform, std::vector<float>* filtered) {
	std::vector<float> fourierRe(FOURIER_PATCH_WIDTH);
	std::vector<float> fourierIm(FOURIER_PATCH_WIDTH);

	interpolate(waveform, &fourierRe, FOURIER_PATCH_WIDTH);

	FFT(1, FOURIER_PATCH_WIDTH_POW, &fourierRe[0], &fourierIm[0]);

	for (size_t i = 0; i < BAND_PASS_HIGH; i++) {
		size_t midpoint = FOURIER_PATCH_WIDTH / 2;
		fourierRe[midpoint + i + 1] = 0;
		fourierIm[midpoint + i + 1] = 0;
		fourierRe[midpoint - i] = 0;
		fourierIm[midpoint - i] = 0;
	}
	for (size_t i = BAND_PASS_LOW; i < FOURIER_PATCH_WIDTH / 2; i++) {
		size_t midpoint = FOURIER_PATCH_WIDTH / 2;
		fourierRe[midpoint + i] = 0;
		fourierIm[midpoint + i] = 0;
		fourierRe[midpoint - i - 1] = 0;
		fourierIm[midpoint - i - 1] = 0;
	}

	FFT(-1, FOURIER_PATCH_WIDTH_POW, &fourierRe[0], &fourierIm[0]);

	filtered->resize(waveform->size());
	interpolate(&fourierRe, filtered, FOURIER_PATCH_WIDTH);
}