#pragma once
#include "cpucompute.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <list>
#include <algorithm>

/*
#define savestring "weights/"
#define datastring "rawdata/"
*/
#define savestring "D:/momCPUTest/weights/"
#define datastring "D:/momCPUTest/rawdata/"

extern std::vector<Layer*> weightlayers;
extern size_t NUM_INPUTS;
extern std::vector<float> outputStdevScale;
extern std::vector<float> outputMeanScale;

float mean(std::vector<float> in);
float stdev(std::vector<float> in, float mean);
void throwError(std::string err);
void loadArchitecture(std::string archfname);
bool loadWeights(std::vector<Layer*> layers, std::string fname);
bool loadBatchNormData(std::vector<Layer*> layers, std::string fname);
bool loadOutputScalings(std::string fname);