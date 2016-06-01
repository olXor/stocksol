#ifndef MOMRUN_HEADER
#define MOMRUN_HEADER

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include "params.h"
#include "kernel.cuh"
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>
#include <memory>
#include <Shlwapi.h>
#include <list>

struct IOPair {
	std::vector<float> inputs;
	float correctoutput;
	size_t samplenum;
};

struct LayerCollection {
	std::vector<ConvolutionMatrices> convMat;
	std::vector<ConvolutionParameters> convPars;
	std::vector<MaxPoolMatrices> mpMat;
	std::vector<MaxPoolParameters> mpPars;
	std::vector<FixedNetMatrices> fixedMat;
	std::vector<FixedNetParameters> fixedPars;

	std::vector<ConvolutionMatrices*> d_convMat;
	std::vector<ConvolutionParameters*> d_convPars;
	std::vector<MaxPoolMatrices*> d_mpMat;
	std::vector<MaxPoolParameters*> d_mpPars;
	std::vector<FixedNetMatrices*> d_fixedMat;
	std::vector<FixedNetParameters*> d_fixedPars;

	size_t numConvolutions;
	size_t numFixedNets;

	//on device:
	float* stepfactor;
	float* correctoutput;
};

void setStrings(std::string data, std::string save);
void freeMemory();
size_t readTrainSet(std::string learnsetname, size_t begin = 1, size_t numIOs = 0);
float runSim(LayerCollection layers, bool train, float customStepFactor, size_t samples = 0, bool print = false);
float testSim(LayerCollection layers, std::string ofname = "");
void saveWeights(LayerCollection layers, std::string fname);
void loadWeights(LayerCollection layers, std::string fname);

void initializeLayers(LayerCollection* layers);
void initializeConvolutionMatrices(ConvolutionMatrices* mat, ConvolutionParameters* pars);
void initializeMPMatrices(MaxPoolMatrices* mat, MaxPoolParameters* pars);
void initializeFixedMatrices(FixedNetMatrices* mat, FixedNetParameters* pars, bool last);
void copyLayersToDevice(LayerCollection* layers);

void calculate(LayerCollection layers, cudaStream_t stream = 0);
void backPropagate(LayerCollection layers, cudaStream_t stream = 0);
#ifdef BATCH_MODE
void batchUpdate(LayerCollection layers);
#endif
float calculateSingleOutput(LayerCollection layers, std::vector<float> inputs);

LayerCollection createLayerCollection();

void randomizeTrainSet(size_t maxIndex = 0);

size_t readExplicitTrainSet(std::string learnsetname, size_t begin, size_t numIOs);
void saveExplicitTrainSet(std::string learnsetname);

void loadParameters(std::string parName);

void sampleReadTrainSet(std::string learnsetname, bool discard, size_t* numDiscards);
float sampleTestSim(LayerCollection layers, std::string ofname, bool testPrintSampleAll = false);

#endif