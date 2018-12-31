#include "cpufeature.h"

void normalizeInputs(float* inputs) {
	//first normalize by standard devation
	float minInput = 9999;
	float maxInput = -9999;
	float stdev = 0;
	float avg = 0;
	for (size_t in = 0; in < NUM_INPUTS; in++) {
		float input = inputs[in];
		stdev += input*input;
		avg += input;
	}
	stdev /= NUM_INPUTS;
	avg /= NUM_INPUTS;
	stdev = sqrt(stdev - avg*avg);

	for (size_t in = 0; in < NUM_INPUTS; in++) {
		inputs[in] = (inputs[in] - avg) / stdev;
	}
}

void normalizeFeatures(std::vector<float>* secondaryFeatures, std::vector<float>* featureMeans, std::vector<float>* featureStdevs, std::vector<bool>* globalScaleMasks) {
	size_t numFeatures = secondaryFeatures->size() / NUM_INPUTS;
	//feature normalization
	for (size_t f = 0; f < numFeatures; f++) {
		if ((*globalScaleMasks)[f]) {
			for (size_t i = 0; i < NUM_INPUTS; i++) {
				(*secondaryFeatures)[f + i * numFeatures] = ((*secondaryFeatures)[f + i * numFeatures] - (*featureMeans)[f]) / (*featureStdevs)[f];
			}
		}
	}
}

void createScaleMasks(std::vector<bool>* globalScaleMask, std::vector<bool>* localScaleMask) {
	globalScaleMask->clear();
	localScaleMask->clear();
	if (CONV_PEAK_FEATURES_INCLUDE_WAVEFORM) {
		globalScaleMask->push_back(false);
		localScaleMask->push_back(false);
	}

	if (CONV_PEAK_FEATURES_INCLUDE_ALL_SLOPES) {
		for (size_t i = 0; i < 4; i++) {
			globalScaleMask->push_back(true);
			localScaleMask->push_back(false);
		}
	}

	if (CONV_PEAK_FEATURES_INCLUDE_NEAR_SLOPE) {
		for (size_t i = 0; i < 2; i++) {
			globalScaleMask->push_back(true);
			localScaleMask->push_back(false);
		}
	}

	if (CONV_PEAK_FEATURES_INCLUDE_FAR_SLOPE) {
		for (size_t i = 0; i < 2; i++) {
			globalScaleMask->push_back(true);
			localScaleMask->push_back(false);
		}
	}

	if (CONV_PEAK_FEATURES_INCLUDE_X_POS) {
		for (size_t i = 0; i < 2; i++) {
			globalScaleMask->push_back(true);
			localScaleMask->push_back(false);
		}
	}

	if (CONV_PEAK_FEATURES_INCLUDE_Y_POS) {
		for (size_t i = 0; i < 2; i++) {
			globalScaleMask->push_back(true);
			localScaleMask->push_back(false);
		}
	}

	if (CONV_PEAK_FEATURES_INCLUDE_SLOPE_DIFF) {
		globalScaleMask->push_back(true);
		localScaleMask->push_back(false);
	}

	if (CONV_PEAK_FEATURES_INCLUDE_FIRST_DERIV) {
		globalScaleMask->push_back(false);
		localScaleMask->push_back(true);
	}

	if (CONV_PEAK_FEATURES_INCLUDE_SECOND_DERIV) {
		globalScaleMask->push_back(false);
		localScaleMask->push_back(true);
	}

	if (CONV_PEAK_FEATURES_INCLUDE_NORM_BY_CENTER) {
		globalScaleMask->push_back(false);
		localScaleMask->push_back(false);
	}

	if (CONV_PEAK_FEATURES_INCLUDE_ALL_DERIV2_SLOPES) {
		for (size_t i = 0; i < 4; i++) {
			globalScaleMask->push_back(true);
			localScaleMask->push_back(false);
		}
	}
}

size_t createSecondaryFeatures(float* inputs, float* peaks, std::vector<float>* secondaryFeatures, std::vector<bool>* localScaleMask) {
	size_t numFeatures = (CONV_PEAK_FEATURES_INCLUDE_ALL_SLOPES ? 4 : 0) + (CONV_PEAK_FEATURES_INCLUDE_NEAR_SLOPE ? 2 : 0) + (CONV_PEAK_FEATURES_INCLUDE_FAR_SLOPE ? 2 : 0) + (CONV_PEAK_FEATURES_INCLUDE_X_POS ? 2 : 0) + (CONV_PEAK_FEATURES_INCLUDE_Y_POS ? 2 : 0) + (CONV_PEAK_FEATURES_INCLUDE_WAVEFORM ? 1 : 0) + (CONV_PEAK_FEATURES_INCLUDE_SLOPE_DIFF ? 1 : 0) + (CONV_PEAK_FEATURES_INCLUDE_FIRST_DERIV ? 1 : 0) + (CONV_PEAK_FEATURES_INCLUDE_SECOND_DERIV ? 1 : 0) + (CONV_PEAK_FEATURES_INCLUDE_NORM_BY_CENTER ? 1 : 0) + (CONV_PEAK_FEATURES_INCLUDE_ALL_DERIV2_SLOPES ? 4 : 0);
	secondaryFeatures->clear();

	std::vector<size_t> peakLocs;
	std::vector<size_t> valleyLocs;
	float minVal = 9999;
	float maxVal = -9999;

	for (size_t i = 0; i < NUM_INPUTS; i++) {
		if (peaks[i] > 0)
			peakLocs.push_back(i);
		else if (peaks[i] < 0)
			valleyLocs.push_back(i);
		minVal = std::min(minVal, inputs[i]);
		maxVal = std::max(maxVal, inputs[i]);
	}
	float minPeakDistance = 9999;
	float minValleyDistance = 9999;
	size_t centerPeak = 0;
	size_t centerValley = 0;
	for (size_t i = 0; i < peakLocs.size(); i++) {
		int dist = abs((int)peakLocs[i] - NUM_INPUTS / 2);
		if (dist < minPeakDistance) {
			minPeakDistance = dist;
			centerPeak = peakLocs[i];
		}
	}
	for (size_t i = 0; i < valleyLocs.size(); i++) {
		int dist = abs((int)valleyLocs[i] - NUM_INPUTS / 2);
		if (dist < minValleyDistance) {
			minValleyDistance = dist;
			centerValley = valleyLocs[i];
		}
	}

#ifdef CONV_PEAK_FEATURES_REJECT_LOW_PEAK_WAVEFORMS
	if (peakLocs.size() < 2 || valleyLocs.size() < 2) {
		/*
		std::cout << "Found pulse with less than 2 peaks or valleys detected" << std::endl;
		for (size_t i = 0; i < NUM_INPUTS; i++) {
		std::cout << inputs[i] << " ";
		}
		std::cout << std::endl;
		for (size_t i = 0; i < NUM_INPUTS; i++) {
		std::cout << peaks[i] << " ";
		}
		std::cout << std::endl;
		*/
		for (size_t i = 0; i < NUM_INPUTS; i++) {
			for (size_t f = 0; f < numFeatures; f++) {
				(*secondaryFeatures).push_back(0);
			}
		}
		return numFeatures;
	}
#endif

	std::vector<float> waveform(NUM_INPUTS);
	std::vector<float> deriv1;
	std::vector<float> deriv2;

	for (size_t w = 0; w < NUM_INPUTS; w++)
		waveform[w] = inputs[w];

	if (CONV_PEAK_FEATURES_INCLUDE_FIRST_DERIV || CONV_PEAK_FEATURES_INCLUDE_SECOND_DERIV || CONV_PEAK_FEATURES_INCLUDE_ALL_DERIV2_SLOPES) {
		computeDerivative(&waveform, &deriv1, DERIV_SMOOTHING_RANGE);
	}

	if (CONV_PEAK_FEATURES_INCLUDE_SECOND_DERIV || CONV_PEAK_FEATURES_INCLUDE_ALL_DERIV2_SLOPES) {
		computeDerivative(&deriv1, &deriv2, DERIV_SMOOTHING_RANGE);
	}

	float maxDeriv1 = -99999;
	float minDeriv1 = 99999;
	float maxDeriv2 = -99999;
	float minDeriv2 = 99999;

	for (size_t i = 0; i < NUM_INPUTS; i++) {
		minDeriv1 = std::min(minDeriv1, deriv1[i]);
		maxDeriv1 = std::max(maxDeriv1, deriv1[i]);
		minDeriv2 = std::min(minDeriv2, deriv2[i]);
		maxDeriv2 = std::max(maxDeriv2, deriv2[i]);
	}

	size_t lastPeak = 0;
	size_t lastValley = 0;

	for (size_t i = 0; i < NUM_INPUTS; i++) {
		bool hasBackPeak = i > peakLocs[0];
		bool hasForwardPeak = i < peakLocs[peakLocs.size() - 1];
		bool hasBackValley = i > valleyLocs[0];
		bool hasForwardValley = i < valleyLocs[valleyLocs.size() - 1];

		if (lastPeak < peakLocs.size() - 1 && i > peakLocs[lastPeak + 1])
			lastPeak++;
		if (lastValley < valleyLocs.size() - 1 && i > valleyLocs[lastValley + 1])
			lastValley++;

		float pulseWidth = (lastPeak < peakLocs.size() - 1 ? 1.0f*(peakLocs[lastPeak + 1] - peakLocs[lastPeak]) : 1.0f*(peakLocs[lastPeak] - peakLocs[lastPeak - 1]));
		float peakXDiffBack;
		float peakYDiffBack;
		if (i > peakLocs[0]) {
			peakXDiffBack = 1.0f*(i - peakLocs[lastPeak]) / pulseWidth;
			peakYDiffBack = (maxVal > minVal ? (inputs[i] - inputs[peakLocs[lastPeak]]) / (maxVal - minVal) : 0.0f);
		}
		else {
			peakXDiffBack = 0;
			peakYDiffBack = 0;
		}

		float peakXDiffFor;
		float peakYDiffFor;
		if (i < peakLocs[peakLocs.size() - 1]) {
			peakXDiffFor = 1.0f*(peakLocs[lastPeak + 1] - i) / pulseWidth;
			peakYDiffFor = (maxVal > minVal ? (inputs[peakLocs[lastPeak + 1]] - inputs[i]) / (maxVal - minVal) : 0.0f);
		}
		else {
			peakXDiffFor = 0;
			peakYDiffFor = 0;
		}

		float valleyXDiffBack;
		float valleyYDiffBack;
		if (i > valleyLocs[0]) {
			valleyXDiffBack = 1.0f*(i - valleyLocs[lastValley]) / pulseWidth;
			valleyYDiffBack = (maxVal > minVal ? (inputs[i] - inputs[valleyLocs[lastValley]]) / (maxVal - minVal) : 0.0f);
		}
		else {
			valleyXDiffBack = 0;
			valleyYDiffBack = 0;
		}


		float valleyXDiffFor;
		float valleyYDiffFor;
		if (i < valleyLocs[valleyLocs.size() - 1]) {
			valleyXDiffFor = 1.0f*(valleyLocs[lastValley + 1] - i) / pulseWidth;
			valleyYDiffFor = (maxVal > minVal ? (inputs[valleyLocs[lastValley + 1]] - inputs[i]) / (maxVal - minVal) : 0.0f);
		}
		else {
			valleyXDiffFor = 0;
			valleyYDiffFor = 0;
		}

		float peakBackHypo = sqrt(peakXDiffBack*peakXDiffBack + peakYDiffBack*peakYDiffBack);
		float peakForHypo = sqrt(peakXDiffFor*peakXDiffFor + peakYDiffFor*peakYDiffFor);
		float valleyBackHypo = sqrt(valleyXDiffBack*valleyXDiffBack + valleyYDiffBack*valleyYDiffBack);
		float valleyForHypo = sqrt(valleyXDiffFor*valleyXDiffFor + valleyYDiffFor*valleyYDiffFor);

		if (CONV_PEAK_FEATURES_INCLUDE_WAVEFORM) {
			secondaryFeatures->push_back(inputs[i]);
		}

		if (CONV_PEAK_FEATURES_INCLUDE_ALL_SLOPES) {
			if (peakBackHypo != 0)
				secondaryFeatures->push_back(peakYDiffBack / peakBackHypo);
			else
				secondaryFeatures->push_back(0.0f);

			if (peakForHypo != 0)
				secondaryFeatures->push_back(peakYDiffFor / peakForHypo);
			else
				secondaryFeatures->push_back(0.0f);

			if (valleyBackHypo != 0)
				secondaryFeatures->push_back(valleyYDiffBack / valleyBackHypo);
			else
				secondaryFeatures->push_back(0.0f);

			if (valleyForHypo != 0)
				secondaryFeatures->push_back(valleyYDiffFor / valleyForHypo);
			else
				secondaryFeatures->push_back(0.0f);
		}

		if (CONV_PEAK_FEATURES_INCLUDE_NEAR_SLOPE) {
			if (i > peakLocs[0] && (i <= valleyLocs[0] || peakLocs[lastPeak] > valleyLocs[lastValley]) && peakBackHypo != 0)
				secondaryFeatures->push_back(peakYDiffBack / peakBackHypo);
			else if (i > valleyLocs[0] && (i <= peakLocs[0] || valleyLocs[lastValley] > peakLocs[lastPeak]) && valleyBackHypo != 0)
				secondaryFeatures->push_back(valleyYDiffBack / valleyBackHypo);
			else
				secondaryFeatures->push_back(0);

			if (i < peakLocs[peakLocs.size() - 1] && (i >= valleyLocs[valleyLocs.size() - 1] || peakLocs[lastPeak + 1] < valleyLocs[lastValley + 1]) && peakForHypo != 0)
				secondaryFeatures->push_back(peakYDiffFor / peakForHypo);
			else if (i < valleyLocs[valleyLocs.size() - 1] && (i >= peakLocs[peakLocs.size() - 1] || valleyLocs[lastValley + 1] < peakLocs[lastPeak + 1]) && valleyForHypo != 0)
				secondaryFeatures->push_back(valleyYDiffFor / valleyForHypo);
			else
				secondaryFeatures->push_back(0);
		}

		if (CONV_PEAK_FEATURES_INCLUDE_FAR_SLOPE) {
			if (i > peakLocs[0] && i > valleyLocs[0] && (peakLocs[lastPeak] < valleyLocs[lastValley]) && peakBackHypo != 0)
				secondaryFeatures->push_back(peakYDiffBack / peakBackHypo);
			else if (i > peakLocs[0] && i > valleyLocs[0] && (valleyLocs[lastValley] < peakLocs[lastPeak]) && valleyBackHypo != 0)
				secondaryFeatures->push_back(valleyYDiffBack / valleyBackHypo);
			else
				secondaryFeatures->push_back(0);

			if (i < peakLocs[peakLocs.size() - 1] && i < valleyLocs[valleyLocs.size() - 1] && (peakLocs[lastPeak + 1] > valleyLocs[lastValley + 1]) && peakForHypo != 0)
				secondaryFeatures->push_back(peakYDiffFor / peakForHypo);
			else if (i < peakLocs[peakLocs.size() - 1] && i < valleyLocs[valleyLocs.size() - 1] && (valleyLocs[lastValley + 1] > peakLocs[lastPeak + 1]) && valleyForHypo != 0)
				secondaryFeatures->push_back(valleyYDiffFor / valleyForHypo);
			else
				secondaryFeatures->push_back(0);
		}

		if (CONV_PEAK_FEATURES_INCLUDE_X_POS) {
			//peakXBack
			if (i > peakLocs[0])
				secondaryFeatures->push_back(i - peakLocs[lastPeak]);
			else
				secondaryFeatures->push_back(0);

			//peakXFor
			if (i < peakLocs[peakLocs.size() - 1])
				secondaryFeatures->push_back(peakLocs[lastPeak + 1] - i);
			else
				secondaryFeatures->push_back(0);
		}

		if (CONV_PEAK_FEATURES_INCLUDE_Y_POS) {
			//peakY
			if (i <= peakLocs[0])
				secondaryFeatures->push_back(inputs[peakLocs[0]] - inputs[i]);
			else if (i >= peakLocs[peakLocs.size() - 1])
				secondaryFeatures->push_back(inputs[peakLocs[peakLocs.size() - 1]] - inputs[i]);
			else
				secondaryFeatures->push_back((inputs[peakLocs[lastPeak]] + inputs[peakLocs[lastPeak + 1]]) / 2 - inputs[i]);

			//valleyY
			if (i <= valleyLocs[0])
				secondaryFeatures->push_back(inputs[valleyLocs[0]] - inputs[i]);
			else if (i >= valleyLocs[valleyLocs.size() - 1])
				secondaryFeatures->push_back(inputs[valleyLocs[valleyLocs.size() - 1]] - inputs[i]);
			else
				secondaryFeatures->push_back((inputs[valleyLocs[lastValley]] + inputs[valleyLocs[lastValley + 1]]) / 2 - inputs[i]);
		}

		if (CONV_PEAK_FEATURES_INCLUDE_SLOPE_DIFF) {
			size_t backPoint = std::max((i > peakLocs[0] ? peakLocs[lastPeak] : 0), (i > valleyLocs[0] ? valleyLocs[lastValley] : 0));
			size_t forwardPoint = std::min((i <= peakLocs[peakLocs.size() - 1] ? (i > peakLocs[0] ? peakLocs[lastPeak + 1] : peakLocs[0]) : 9999), (i <= valleyLocs[valleyLocs.size() - 1] ? (i > valleyLocs[0] ? valleyLocs[lastValley + 1] : valleyLocs[0]) : 9999));
			if (backPoint == 0 || forwardPoint == 9999 || forwardPoint <= backPoint || inputs[forwardPoint] == inputs[backPoint]) {
				secondaryFeatures->push_back(0);
			}
			else {
				float slopeHeight = inputs[backPoint] + (inputs[forwardPoint] - inputs[backPoint]) * (i - backPoint) / (forwardPoint - backPoint);
				secondaryFeatures->push_back((inputs[i] - slopeHeight) / fabs(inputs[forwardPoint] - inputs[backPoint]));
			}
		}

		if (CONV_PEAK_FEATURES_INCLUDE_FIRST_DERIV) {
			secondaryFeatures->push_back(deriv1[i]);
		}

		if (CONV_PEAK_FEATURES_INCLUDE_SECOND_DERIV) {
			secondaryFeatures->push_back(deriv2[i]);
		}

		if (CONV_PEAK_FEATURES_INCLUDE_NORM_BY_CENTER) {
			if (inputs[centerPeak] > inputs[centerValley])
				secondaryFeatures->push_back(2.0f*(inputs[i] - inputs[centerValley]) / (inputs[centerPeak] - inputs[centerValley]) - 1.0f);
			else
				secondaryFeatures->push_back(0);
		}

		if (CONV_PEAK_FEATURES_INCLUDE_ALL_DERIV2_SLOPES) {
			float derivPeakYDiffBack;
			if (i > peakLocs[0]) {
				derivPeakYDiffBack = (maxDeriv2 > minDeriv2 ? (deriv2[i] - deriv2[peakLocs[lastPeak]]) / (maxDeriv2 - minDeriv2) : 0.0f);
			}
			else {
				derivPeakYDiffBack = 0;
			}

			float derivPeakYDiffFor;
			if (i < peakLocs[peakLocs.size() - 1]) {
				derivPeakYDiffFor = (maxDeriv2 > minDeriv2 ? (deriv2[peakLocs[lastPeak + 1]] - deriv2[i]) / (maxDeriv2 - minDeriv2) : 0.0f);
			}
			else {
				derivPeakYDiffFor = 0;
			}

			float derivValleyYDiffBack;
			if (i > valleyLocs[0]) {
				derivValleyYDiffBack = (maxDeriv2 > minDeriv2 ? (deriv2[i] - deriv2[valleyLocs[lastValley]]) / (maxDeriv2 - minDeriv2) : 0.0f);
			}
			else {
				derivValleyYDiffBack = 0;
			}


			float derivValleyYDiffFor;
			if (i < valleyLocs[valleyLocs.size() - 1]) {
				derivValleyYDiffFor = (maxDeriv2 > minDeriv2 ? (deriv2[valleyLocs[lastValley + 1]] - deriv2[i]) / (maxDeriv2 - minDeriv2) : 0.0f);
			}
			else {
				derivValleyYDiffFor = 0;
			}

			float derivPeakBackHypo = sqrt(peakXDiffBack*peakXDiffBack + derivPeakYDiffBack*derivPeakYDiffBack);
			float derivPeakForHypo = sqrt(peakXDiffFor*peakXDiffFor + derivPeakYDiffFor*derivPeakYDiffFor);
			float derivValleyBackHypo = sqrt(valleyXDiffBack*valleyXDiffBack + derivValleyYDiffBack*derivValleyYDiffBack);
			float derivValleyForHypo = sqrt(valleyXDiffFor*valleyXDiffFor + derivValleyYDiffFor*derivValleyYDiffFor);

			if (derivPeakBackHypo != 0)
				secondaryFeatures->push_back(derivPeakYDiffBack / derivPeakBackHypo);
			else
				secondaryFeatures->push_back(0.0f);

			if (derivPeakForHypo != 0)
				secondaryFeatures->push_back(derivPeakYDiffFor / derivPeakForHypo);
			else
				secondaryFeatures->push_back(0.0f);

			if (derivValleyBackHypo != 0)
				secondaryFeatures->push_back(derivValleyYDiffBack / derivValleyBackHypo);
			else
				secondaryFeatures->push_back(0.0f);

			if (derivValleyForHypo != 0)
				secondaryFeatures->push_back(derivValleyYDiffFor / derivValleyForHypo);
			else
				secondaryFeatures->push_back(0.0f);
		}
	}

	//local scaling
	for (size_t f = 0; f < numFeatures; f++) {
		if (!(*localScaleMask)[f])
			continue;
		float mean = 0;
		float stdev = 0;
		for (size_t i = 0; i < NUM_INPUTS; i++) {
			float val = (*secondaryFeatures)[f + i*numFeatures];
			mean += val;
			stdev += val*val;
		}
		mean /= NUM_INPUTS;
		stdev /= NUM_INPUTS;
		stdev = sqrt(stdev - mean*mean);
		for (size_t i = 0; i < NUM_INPUTS; i++) {
			(*secondaryFeatures)[f + i*numFeatures] = (stdev > 0 ? ((*secondaryFeatures)[f + i*numFeatures] - mean) / stdev : 0);
		}
	}

	return secondaryFeatures->size() / NUM_INPUTS;
}

size_t processInput(float* inputs, float* peaks, std::vector<float>* secondaryFeatures, std::vector<float>* featureMeans, std::vector<float>* featureStdevs) {
	normalizeInputs(inputs);
	std::vector<bool> globalScaleMasks;
	std::vector<bool> localScaleMasks;
	createScaleMasks(&globalScaleMasks, &localScaleMasks);
	size_t numFeatures = createSecondaryFeatures(inputs, peaks, secondaryFeatures, &localScaleMasks);
	normalizeFeatures(secondaryFeatures, featureMeans, featureStdevs, &globalScaleMasks);
	return numFeatures;
}

bool loadFeatureNorms(std::string fname, std::vector<float>* featureMeans, std::vector<float>* featureStdevs) {
	featureMeans->clear();
	featureStdevs->clear();

	std::ifstream featurefile(fname);
	if (!featurefile.is_open())
		return false;

	std::string line;
	std::getline(featurefile, line);
	std::stringstream lss(line);
	float val;
	while (lss >> val)
		featureMeans->push_back(val);

	std::getline(featurefile, line);
	std::stringstream lss2(line);
	while (lss2 >> val)
		featureStdevs->push_back(val);

	return true;
}
