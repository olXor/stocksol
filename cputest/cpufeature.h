#pragma once
#include "cpucommon.h"

#define NUM_INPUTS 512

#define USE_EXTRA_CONV_PEAK_FEATURES 1
#define CONV_PEAK_FEATURES_INCLUDE_WAVEFORM true
#define CONV_PEAK_FEATURES_INCLUDE_ALL_SLOPES true
#define CONV_PEAK_FEATURES_INCLUDE_REL_SLOPE false
#define CONV_PEAK_FEATURES_INCLUDE_X_POS false
#define CONV_PEAK_FEATURES_INCLUDE_Y_POS false
#define CONV_PEAK_FEATURES_INCLUDE_SLOPE_DIFF false
#define CONV_PEAK_FEATURES_INCLUDE_FIRST_DERIV false
#define CONV_PEAK_FEATURES_INCLUDE_SECOND_DERIV false

size_t createSecondaryFeatures(float* inputs, float* peaks, std::vector<float>* secondaryFeatures);
