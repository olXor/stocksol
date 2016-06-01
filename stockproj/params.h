#ifndef PARAMS_HEADER
#define PARAMS_HEADER

//#define BATCH_MODE

extern size_t NUM_INPUTS;
#define NUM_NEURONS 32

#define TRANSFER_FUNCTION_LIMIT 50.0f
#define TRANSFER_WIDTH 1.0f
#ifdef BATCH_MODE
#define MAX_WEIGHT_CHANGE 0.1f
#else
#define MAX_WEIGHT_CHANGE 0.1f
#endif

#define STEPFACTOR 1e-4f

//#define LOCAL

extern float INITIAL_OUTPUT_AVERAGE;

extern bool tradeTypeLong;

extern float OUTPUT_DIVISOR;

extern std::string savename;
extern std::string trainstring;
extern std::string randtrainstring;

#endif