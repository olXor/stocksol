#ifndef PARAMS_HEADER
#define PARAMS_HEADER

//#define NUM_INPUTS 64
extern size_t NUM_INPUTS;
#define NUM_NEURONS 32

#define TRANSFER_FUNCTION_LIMIT 50.0f
#define TRANSFER_WIDTH 1.0f

//#define STEPFACTOR 0.000001f
#define STEPFACTOR 1e-4f
//#define MAX_STEP 5e-3f

//#define INITIAL_OUTPUT_THRESHOLD 0.0f
extern float INITIAL_OUTPUT_THRESHOLD;

#define TRADE_TYPE_LONG 1
#define TRADE_TYPE_SHORT 0

#define TRADE_TYPE 	TRADE_TYPE_LONG

//#define OUTPUT_DIVISOR 50.0f
extern float OUTPUT_DIVISOR;

//#define savename "randsample"
//#define savename "badfast"
extern std::string savename;
//#define trainstring "2011-2014"
extern std::string trainstring;
//#define randtrainstring "randomlong"
extern std::string randtrainstring;

#endif