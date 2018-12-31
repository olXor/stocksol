#pragma once
#include "cpucommon.h"
#include "cpuinterpolate.h"

#define NUM_INPUTS 512

#define DERIV_SMOOTHING_RANGE 5

#define CONV_PEAK_FEATURES_REJECT_LOW_PEAK_WAVEFORMS
#define USE_EXTRA_CONV_PEAK_FEATURES 1
#define CONV_PEAK_FEATURES_INCLUDE_WAVEFORM true
#define CONV_PEAK_FEATURES_INCLUDE_ALL_SLOPES false
#define CONV_PEAK_FEATURES_INCLUDE_NEAR_SLOPE false
#define CONV_PEAK_FEATURES_INCLUDE_FAR_SLOPE false
#define CONV_PEAK_FEATURES_INCLUDE_X_POS false
#define CONV_PEAK_FEATURES_INCLUDE_Y_POS false
#define CONV_PEAK_FEATURES_INCLUDE_SLOPE_DIFF false
#define CONV_PEAK_FEATURES_INCLUDE_FIRST_DERIV false
#define CONV_PEAK_FEATURES_INCLUDE_SECOND_DERIV false
#define CONV_PEAK_FEATURES_INCLUDE_NORM_BY_CENTER false
#define CONV_PEAK_FEATURES_INCLUDE_ALL_DERIV2_SLOPES true

size_t processInput(float* inputs, float* peaks, std::vector<float>* secondaryFeatures, std::vector<float>* featureMeans, std::vector<float>* featureStdevs);
bool loadFeatureNorms(std::string fname, std::vector<float>* featureMeans, std::vector<float>* featureStdevs);
