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

#define FULL_NETWORK 0
#define CONV_ONLY 1
#define FIXED_ONLY 2

struct IOPair {
	std::vector<float> inputs;
	float correctoutput;
	float secondaryoutput;
	std::vector<float> correctbins;
	std::vector<float> secondarybins;
	size_t samplenum;
	float weight = 1.0f;
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

struct PairedConvCollection {
	LayerCollection conv1;
	LayerCollection conv2;
	LayerCollection fixed;
};

void setStrings(std::string data, std::string save);
void freeMemory();
size_t readTrainSet(std::string learnsetname, size_t begin = 1, size_t numIOs = 0, bool overrideBinningSwitch = false, bool runOnTestSet = false);
float runSim(LayerCollection layers, bool train, float customStepFactor, size_t samples = 0, bool print = false, float* secondaryError = NULL, bool runOnTestSet = false, float* outAverages = NULL, std::ofstream* outfile = NULL, bool updateBinFreqs = false);
float runPairedSim(PairedConvCollection layers, bool train, float customStepFactor, size_t samples = 0, bool print = false, size_t pairsAveraged = 0);
float testSim(LayerCollection layers, std::string ofname = "");
void saveWeights(LayerCollection layers, std::string fname);
bool loadWeights(LayerCollection layers, std::string fname);
bool loadPairedWeights(PairedConvCollection layers, std::string savename);
void savePairedWeights(PairedConvCollection layers, std::string savename);

int getLCType();

void initializeLayers(LayerCollection* layers);
void initializeConvolutionMatrices(ConvolutionMatrices* mat, ConvolutionParameters* pars);
void initializeMPMatrices(MaxPoolMatrices* mat, MaxPoolParameters* pars);
void initializeFixedMatrices(FixedNetMatrices* mat, FixedNetParameters* pars, bool last);
void copyLayersToDevice(LayerCollection* layers);
void freeDeviceLayers(LayerCollection* layers);

void calculate(LayerCollection layers, cudaStream_t stream = 0);
void backPropagate(LayerCollection layers, float stepfactor, cudaStream_t stream = 0);
#ifdef BATCH_MODE
void batchUpdate(LayerCollection layers);
#endif
float calculateSingleOutput(LayerCollection layers, std::vector<float> inputs);

LayerCollection createLayerCollection(size_t numInputs = 0, int LCType = FULL_NETWORK);
PairedConvCollection createAndInitializePairedConvCollection(size_t numInputs);

void randomizeTrainSet(size_t maxIndex = 0);

size_t readExplicitTrainSet(std::string learnsetname, size_t begin, size_t numIOs, bool runOnTestSet = false);
void saveExplicitTrainSet(std::string learnsetname);

void loadParameters(std::string parName);

void sampleReadTrainSet(std::string learnsetname, bool discard, size_t* numDiscards, bool overrideBinningSwitch = false, bool runOnTestSet = false, size_t startingSample = 0, bool readInBlocks = false);
float sampleTestSim(LayerCollection layers, std::ofstream* outfile, bool testPrintSampleAll = false, bool print = true, bool runOnTestSet = false);

std::vector<float> getBinnedOutput(float output);

void initializeDropoutRNG(LayerCollection* lc);
void disableDropout();
void enableDropout();
void generateDropoutMask(LayerCollection* lc);

std::vector<std::vector<IOPair>> getBinnedTrainset();

std::vector<IOPair>* getTrainSet();
std::vector<IOPair>* getTestSet();

size_t readTwoPriceTrainSet(std::string learnsetname, size_t begin, size_t numIOs, bool overrideBinningSwitch = false, bool runOnTestSet = false);

bool discardInput(float* inputs);

float mean(std::vector<float> in);
float stdev(std::vector<float> in, float mean);

void markTime();
long long getTimeSinceMark();

void initializeDOutAverages();
void generateTrainWeightBins();
void fillTrainsetIndicesByBin();
size_t getTrainsetIndexByBinFrequency();

#endif