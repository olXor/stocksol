#ifndef PARAMS_HEADER
#define PARAMS_HEADER

#include <vector>

#define BATCH_MODE

extern size_t NUM_INPUTS;
#define NUM_NEURONS 32

#define TRANSFER_FUNCTION_LIMIT 50.0f
#define TRANSFER_WIDTH 1.0f
#ifdef BATCH_MODE
#define MAX_WEIGHT_CHANGE 0.1f
#else
#define MAX_WEIGHT_CHANGE 0.1f
#endif

#define TRANSFER_TYPE_RECTIFIER 1
#define TRANSFER_TYPE_SIGMOID 2

#define STEPFACTOR 1e-4f

#define LOCAL

extern float INITIAL_OUTPUT_AVERAGE;

extern bool tradeTypeLong;

extern float OUTPUT_DIVISOR;

extern size_t numBins;
extern float binMin;
extern float binMax;
extern float binWidth;

extern std::string savename;
extern std::string trainstring;
extern std::string randtrainstring;

extern size_t pairProximity;

extern std::vector<float> testSelectBinMins;
extern std::vector<float> testSelectBinMaxes;
extern std::vector<float> oppositeSelectBinMins;
extern std::vector<float> oppositeSelectBinMaxes;

#define BIN_POSITIVE_OUTPUT 100.0f
#define BIN_NEGATIVE_OUTPUT 0.0f

#define INTERVALS_PER_DATASET 100
#endif