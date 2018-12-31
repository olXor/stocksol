#pragma once

#include "math.h"
#include <vector>
#include <algorithm>
#include "cpuinterpolate.h"

size_t findFirstHarmonic(std::vector<float>* freq);
void findPeaksAndValleys(std::vector<float>* waveform, std::vector<float>* peaks);
void applyBandPassFilter(std::vector<float>* waveform, std::vector<float>* filtered);