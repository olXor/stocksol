#include "stockrun.cuh"

std::vector<IOPair> trainset;

std::string datastring;
std::string savestring;

//parameters
size_t NUM_INPUTS = 64;
float INITIAL_OUTPUT_THRESHOLD = 0.0f;
float OUTPUT_DIVISOR = 1.0f;
std::string savename = "weights";
std::string trainstring = "trainset";
std::string randtrainstring = "rantrainset";

void setStrings(std::string data, std::string save) {
	datastring = data;
	savestring = save;
}

size_t readTrainSet(std::string learnsetname, size_t begin, size_t numIOs) {
	trainset.clear();
	std::stringstream learnsetss;
	learnsetss << datastring << learnsetname;
	std::ifstream learnset(learnsetss.str().c_str());
	std::string line;
	std::list<float> prices;
	size_t ionum = 0;
	while (getline(learnset, line)) {
		std::stringstream lss(line);
		float price;
		float longprof;
		float shortprof;
		lss >> price >> longprof >> shortprof;

		prices.push_back(price);
		if (prices.size() > NUM_INPUTS)
			prices.pop_front();

		if (prices.size() == NUM_INPUTS) {
			ionum++;
			if (numIOs > 0 && (ionum < begin || ionum >= begin + numIOs))
				continue;
			IOPair io;
			io.inputs.resize(NUM_INPUTS);
			if (TRADE_TYPE == TRADE_TYPE_LONG)
				io.correctoutput = longprof/OUTPUT_DIVISOR;
			else
				io.correctoutput = shortprof/OUTPUT_DIVISOR;

			size_t n = 0;
			for (std::list<float>::iterator it = prices.begin(); it != prices.end(); it++) {
				io.inputs[n] = *it;
				n++;
			}

			float maxinput = -999999;
			float mininput = 999999;
			for (size_t j = 0; j < NUM_INPUTS; j++) {
				if (io.inputs[j] > maxinput)
					maxinput = io.inputs[j];
				if (io.inputs[j] < mininput)
					mininput = io.inputs[j];
			}
			for (size_t j = 0; j<NUM_INPUTS; j++) {
				if (maxinput > mininput)
					io.inputs[j] = 2 * (io.inputs[j] - mininput) / (maxinput - mininput) - 1;
				else
					io.inputs[j] = 0;
			}
			trainset.push_back(io);
		}
	}
	return ionum;
}

float runSim(LayerCollection layers, bool train, float customStepFactor, bool print) {
	float* d_inputs;
	if (layers.numConvolutions > 0) {
		if (layers.convPars[0].numInputLocs != NUM_INPUTS || layers.convPars[0].numInputNeurons != 1)
			throw std::runtime_error("inputs to first layer don't match data set");
		d_inputs = layers.convMat[0].inlayer;
	}
	else if (layers.numFixedNets > 0) {
		if (layers.fixedPars[0].numInputNeurons != NUM_INPUTS)
			throw std::runtime_error("inputs to first layer don't match data set");
		d_inputs = layers.fixedMat[0].inlayer;
	}
	else
		throw std::runtime_error("tried to run on a network with no convolutions and no fixed networks");

	float h_output[1];

	float error = 0;
	size_t numerrors = 0;
	for (size_t i = 0; i < trainset.size(); i++) {
		numerrors++;

		//----calculate----
		checkCudaErrors(cudaMemcpy(d_inputs, &trainset[i].inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice));

		calculate(layers);

		checkCudaErrors(cudaMemcpy(h_output, layers.fixedMat[layers.numFixedNets-1].outlayer, sizeof(float), cudaMemcpyDeviceToHost));
		//----end calculate-----

		float newerror = (h_output[0] - trainset[i].correctoutput);
		//error += newerror*newerror;
		error += fabs(newerror);
		if (print) {
			/*
			std::cout << "Inputs: ";
			for (size_t j = 0; j < NUM_INPUTS; j++) {
				std::cout << trainset[i].inputs[j] << " ";
			}
			std::cout << std::endl;
			*/
			std::cout << "Output: " << h_output[0]*OUTPUT_DIVISOR << ", Correct: " << trainset[i].correctoutput*OUTPUT_DIVISOR << ", Error: " << newerror << std::endl;
		}
		if (train) {
			float stepfac = STEPFACTOR*customStepFactor;

			float bperror[1];
			bperror[0] = stepfac*newerror;
#ifdef MAX_STEP
			if (bperror[0] > MAX_STEP)
				bperror[0] = MAX_STEP;
			else if (bperror[0] < -MAX_STEP)
				bperror[0] = -MAX_STEP;
#endif

			//---back propagate----
			checkCudaErrors(cudaMemcpy(layers.fixedMat[layers.numFixedNets - 1].outErrors, bperror, sizeof(float), cudaMemcpyHostToDevice));

			backPropagate(layers);
			//---end back propagate----
		}
	}

	if (numerrors > 0) {
		error /= numerrors;
		//error = sqrt(error);
	}
	return error*OUTPUT_DIVISOR;
}

void saveWeights(LayerCollection layers, std::string fname) {
	std::stringstream fss;
	fss << savestring << fname;
	std::ofstream outfile(fss.str().c_str());

	for (size_t lay = 0; lay < layers.numConvolutions; lay++) {
		ConvolutionMatrices mat = layers.convMat[lay];
		ConvolutionParameters pars = layers.convPars[lay];

		size_t numWeights = pars.numInputNeurons*pars.numOutputNeurons*pars.convSize;
		size_t numThresholds = pars.numOutputNeurons;
		float* h_weights = new float[numWeights];
		float* h_thresholds = new float[numThresholds];
		checkCudaErrors(cudaMemcpy(h_weights, mat.weights, numWeights*sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_thresholds, mat.outThresholds, numThresholds*sizeof(float), cudaMemcpyDeviceToHost));

		outfile << "Convolution Layer " << lay << ": " << std::endl;
		for (size_t i = 0; i < pars.numOutputNeurons; i++) {
			outfile << h_thresholds[i] << " | " << std::endl;
			for (size_t j = 0; j < pars.numInputNeurons; j++) {
				outfile << "     ";
				for (size_t k = 0; k < pars.convSize; k++) {
					outfile << h_weights[j + i*pars.numInputNeurons + k*pars.numInputNeurons*pars.numOutputNeurons] << " ";
				}
				outfile << std::endl;
			}
			outfile << std::endl << std::endl << std::endl;
		}

		delete[] h_weights;
		delete[] h_thresholds;
	}

	for (size_t lay = 0; lay < layers.numFixedNets; lay++) {
		FixedNetMatrices mat = layers.fixedMat[lay];
		FixedNetParameters pars = layers.fixedPars[lay];

		size_t numWeights = pars.numInputNeurons*pars.numOutputNeurons;
		size_t numThresholds = pars.numOutputNeurons;
		float* h_weights = new float[numWeights];
		float* h_thresholds = new float[numThresholds];
		checkCudaErrors(cudaMemcpy(h_weights, mat.weights, numWeights*sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_thresholds, mat.outThresholds, numThresholds*sizeof(float), cudaMemcpyDeviceToHost));

		outfile << "Fixed Layer " << lay << ": " << std::endl;
		for (size_t i = 0; i < pars.numOutputNeurons; i++) {
			outfile << h_thresholds[i] << " | ";
			for (size_t j = 0; j < pars.numInputNeurons; j++) {
					outfile << h_weights[j + i*pars.numInputNeurons] << " ";
			}
			outfile << std::endl;
		}
		outfile << std::endl << std::endl;

		delete[] h_weights;
		delete[] h_thresholds;
	}
}

void loadWeights(LayerCollection layers, std::string fname) {
	std::stringstream fss;
	fss << savestring << fname;

	if (!PathFileExists(fss.str().c_str())) {
		std::cout << "No weights file found; starting with random weights" << std::endl;
		return;
	}

	std::ifstream infile(fss.str().c_str());

	for (size_t lay = 0; lay < layers.numConvolutions; lay++) {
		ConvolutionMatrices mat = layers.convMat[lay];
		ConvolutionParameters pars = layers.convPars[lay];

		size_t numWeights = pars.numInputNeurons*pars.numOutputNeurons*pars.convSize;
		size_t numThresholds = pars.numOutputNeurons;
		float* h_weights = new float[numWeights];
		float* h_thresholds = new float[numThresholds];

		std::string dum;

		infile >> dum >> dum >> dum;	//"Convolution Layer #:"
		for (size_t i = 0; i < pars.numOutputNeurons; i++) {
			infile >> h_thresholds[i] >> dum;	//" | "
			for (size_t j = 0; j < pars.numInputNeurons; j++) {
				for (size_t k = 0; k < pars.convSize; k++) {
					infile >> h_weights[j + i*pars.numInputNeurons + k*pars.numInputNeurons*pars.numOutputNeurons];
				}
			}
		}

		checkCudaErrors(cudaMemcpy(mat.weights, h_weights, numWeights*sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(mat.outThresholds, h_thresholds, numThresholds*sizeof(float), cudaMemcpyHostToDevice));
		delete[] h_weights;
		delete[] h_thresholds;
	}

	for (size_t lay = 0; lay < layers.numFixedNets; lay++) {
		FixedNetMatrices mat = layers.fixedMat[lay];
		FixedNetParameters pars = layers.fixedPars[lay];

		size_t numWeights = pars.numInputNeurons*pars.numOutputNeurons;
		size_t numThresholds = pars.numOutputNeurons;
		float* h_weights = new float[numWeights];
		float* h_thresholds = new float[numThresholds];

		std::string dum;

		infile >> dum >> dum >> dum; //"Fixed Layer #:"
		for (size_t i = 0; i < pars.numOutputNeurons; i++) {
			infile >> h_thresholds[i] >> dum; //" | "
			for (size_t j = 0; j < pars.numInputNeurons; j++) {
					infile >> h_weights[j + i*pars.numInputNeurons];
			}
		}

		checkCudaErrors(cudaMemcpy(mat.weights, h_weights, numWeights*sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(mat.outThresholds, h_thresholds, numThresholds*sizeof(float), cudaMemcpyHostToDevice));

		delete[] h_weights;
		delete[] h_thresholds;
	}
}


void throwLayerLinkError() {
	throw std::runtime_error("Couldn't link network layers correctly");
}

void initializeLayers(LayerCollection* layers){
	if (layers->convPars.size() != layers->numConvolutions)
		throw std::runtime_error("Number of ConvolutionParameters given doesn't match the number of convolutions");

	layers->convMat.resize(layers->numConvolutions);
	for (size_t i = 0; i < layers->numConvolutions; i++) {
		initializeConvolutionMatrices(&layers->convMat[i], &layers->convPars[i]);
	}

	if (layers->mpPars.size() > layers->numConvolutions)
		throw std::runtime_error("More MaxPool layers than convolutions given");

	layers->mpMat.resize(layers->mpPars.size());
	for (size_t i = 0; i < layers->mpPars.size(); i++) {
		initializeMPMatrices(&layers->mpMat[i], &layers->mpPars[i]);
	}

	if (layers->fixedPars.size() != layers->numFixedNets)
		throw std::runtime_error("Number of FixedNetParameters given doesn't match the number of fixed nets");
	layers->fixedMat.resize(layers->numFixedNets);
	for (size_t i = 0; i < layers->numFixedNets; i++) {
		bool last = (i == layers->numFixedNets - 1);
			
		initializeFixedMatrices(&layers->fixedMat[i], &layers->fixedPars[i], last);
	}

	//now link the layers together
	for (size_t i = 0; i < layers->numConvolutions; i++) {
		if (i != 0) {
			if (layers->mpMat.size() > i - 1) {
				if (layers->mpMat[i - 1].numOutputElements != layers->convMat[i].numInputElements)
					throwLayerLinkError();

				checkCudaErrors(cudaFree(layers->convMat[i].inlayer));
				layers->convMat[i].inlayer = layers->mpMat[i - 1].outlayer;
				checkCudaErrors(cudaFree(layers->convMat[i].inErrors));
				layers->convMat[i].inErrors = layers->mpMat[i - 1].outError;
			}
			else {
				if (layers->convMat[i - 1].numOutputElements != layers->convMat[i].numInputElements)
					throwLayerLinkError();

				checkCudaErrors(cudaFree(layers->convMat[i].inlayer));
				layers->convMat[i].inlayer = layers->convMat[i - 1].outlayer;
				checkCudaErrors(cudaFree(layers->convMat[i].inErrors));
				layers->convMat[i].inErrors = layers->convMat[i - 1].outErrors;
			}
		}

		if (layers->mpMat.size() > i) {
			if (layers->convMat[i].numOutputElements != layers->mpMat[i].numInputElements)
				throwLayerLinkError();

			checkCudaErrors(cudaFree(layers->mpMat[i].inlayer));
			layers->mpMat[i].inlayer = layers->convMat[i].outlayer;
			checkCudaErrors(cudaFree(layers->mpMat[i].inError));
			layers->mpMat[i].inError = layers->convMat[i].outErrors;
		}
	}

	for (size_t i = 0; i < layers->numFixedNets; i++) {
		size_t numPrevOutputs;
		float* prevOutLayer;
		float* prevOutErrors;
		if (i == 0 && layers->mpMat.size() < layers->numConvolutions) {
			size_t lastConv = layers->numConvolutions - 1;
			numPrevOutputs = layers->convMat[lastConv].numOutputElements;
			prevOutLayer = layers->convMat[lastConv].outlayer;
			prevOutErrors = layers->convMat[lastConv].outErrors;
		}
		else if (i == 0){
			size_t lastConv = layers->numConvolutions - 1;
			numPrevOutputs = layers->mpMat[lastConv].numOutputElements;
			prevOutLayer = layers->mpMat[lastConv].outlayer;
			prevOutErrors = layers->mpMat[lastConv].outError;
		}
		else {
			numPrevOutputs = layers->fixedMat[i - 1].numOutputElements;
			prevOutLayer = layers->fixedMat[i - 1].outlayer;
			prevOutErrors = layers->fixedMat[i - 1].outErrors;
		}

		if (numPrevOutputs != layers->fixedMat[i].numInputElements)
			throwLayerLinkError();

		FixedNetMatrices* mat = &layers->fixedMat[i];
		checkCudaErrors(cudaFree(mat->inlayer));
		mat->inlayer = prevOutLayer;
		checkCudaErrors(cudaFree(mat->inErrors));
		mat->inErrors = prevOutErrors;
	}

	copyLayersToDevice(layers);
}

void initializeConvolutionMatrices(ConvolutionMatrices* mat, ConvolutionParameters* pars) {
	checkCudaErrors(cudaMalloc(&mat->inlayer, pars->numInputLocs*pars->numInputNeurons*sizeof(float)));
	checkCudaErrors(cudaMalloc(&mat->outlayer, pars->numOutputLocs*pars->numOutputNeurons*sizeof(float)));

	size_t numWeights = pars->numInputNeurons*pars->numOutputNeurons*pars->convSize;
	float* h_weights = new float[numWeights];
	for (size_t i = 0; i < numWeights; i++) {
		h_weights[i] = (rand() % 21 - 10.0f) / 10.0f / (pars->numInputNeurons*pars->convSize);
	}
	checkCudaErrors(cudaMalloc(&mat->weights, numWeights*sizeof(float)));
	checkCudaErrors(cudaMemcpy(mat->weights, h_weights, numWeights*sizeof(float), cudaMemcpyHostToDevice));
	delete[] h_weights;

#ifdef BATCH_MODE
	checkCudaErrors(cudaMalloc(&mat->weightChanges, numWeights*sizeof(float)));
	checkCudaErrors(cudaMemset(mat->weightChanges, 0, numWeights*sizeof(float)));
#endif

	size_t numThresholds = pars->numOutputNeurons;
	float* h_thresholds = new float[numThresholds];
	for (size_t i = 0; i < numThresholds; i++){
		h_thresholds[i] = (rand() % 21 - 10.0f) / 10.0f / (pars->numInputNeurons*pars->convSize);
	}
	checkCudaErrors(cudaMalloc(&mat->outThresholds, numThresholds*sizeof(float)));
	checkCudaErrors(cudaMemcpy(mat->outThresholds, h_thresholds, numThresholds*sizeof(float), cudaMemcpyHostToDevice));
	delete[] h_thresholds;

#ifdef BATCH_MODE
	checkCudaErrors(cudaMalloc(&mat->outThreshChanges, numThresholds*sizeof(float)));
	checkCudaErrors(cudaMemset(mat->outThreshChanges, 0, numThresholds*sizeof(float)));
#endif

	checkCudaErrors(cudaMalloc(&mat->outTDs, pars->numOutputNeurons*pars->numOutputLocs*sizeof(float)));
	checkCudaErrors(cudaMalloc(&mat->inErrors, pars->numInputNeurons*pars->numInputLocs*sizeof(float)));
	checkCudaErrors(cudaMalloc(&mat->outErrors, pars->numOutputNeurons*pars->numOutputLocs*sizeof(float)));

	mat->forwardSharedMem = getConvolveSharedSize(pars);
	mat->backwardSharedMem = getBPConvolutionSharedSize(pars);
	mat->numInputElements = pars->numInputNeurons*pars->numInputLocs;
	mat->numOutputElements = pars->numOutputNeurons*pars->numOutputLocs;
}

void initializeMPMatrices(MaxPoolMatrices* mat, MaxPoolParameters* pars) {
	size_t numInputs = pars->numInputLocs*pars->numNeurons;
	size_t numOutputs = pars->numOutputLocs*pars->numNeurons;
	if (numInputs != 2 * numOutputs)
		throw std::runtime_error("Invalid MaxPool dimensions");

	checkCudaErrors(cudaMalloc(&mat->inlayer, numInputs*sizeof(float)));
	checkCudaErrors(cudaMalloc(&mat->outlayer, numOutputs*sizeof(float)));
	checkCudaErrors(cudaMalloc(&mat->maxIndices, numOutputs*sizeof(size_t)));
	checkCudaErrors(cudaMalloc(&mat->inError, numInputs*sizeof(float)));
	checkCudaErrors(cudaMalloc(&mat->outError, numOutputs*sizeof(float)));

	mat->numInputElements = pars->numInputLocs*pars->numNeurons;
	mat->numOutputElements = pars->numOutputLocs*pars->numNeurons;
}

void initializeFixedMatrices(FixedNetMatrices* mat, FixedNetParameters* pars, bool last) {
	checkCudaErrors(cudaMalloc(&mat->inlayer, pars->numInputNeurons*sizeof(float)));
	checkCudaErrors(cudaMalloc(&mat->outlayer, pars->numOutputNeurons*sizeof(float)));

	size_t numWeights = pars->numInputNeurons*pars->numOutputNeurons;
	float* h_weights = new float[numWeights];
	for (size_t i = 0; i < numWeights; i++) {
		h_weights[i] = (rand() % 21 - 10.0f) / 10.0f / (pars->numInputNeurons);
	}
	checkCudaErrors(cudaMalloc(&mat->weights, numWeights*sizeof(float)));
	checkCudaErrors(cudaMemcpy(mat->weights, h_weights, numWeights*sizeof(float), cudaMemcpyHostToDevice));
	delete[] h_weights;

#ifdef BATCH_MODE
	checkCudaErrors(cudaMalloc(&mat->weightChanges, numWeights*sizeof(float)));
	checkCudaErrors(cudaMemset(mat->weightChanges, 0, numWeights*sizeof(float)));
#endif

	size_t numThresholds = pars->numOutputNeurons;
	float* h_thresholds = new float[numThresholds];
	for (size_t i = 0; i < numThresholds; i++){
		h_thresholds[i] = (rand() % 21 - 10.0f) / 10.0f / (pars->numInputNeurons);
	}

	if (last)
		h_thresholds[0] = INITIAL_OUTPUT_THRESHOLD;

	checkCudaErrors(cudaMalloc(&mat->outThresholds, numThresholds*sizeof(float)));
	checkCudaErrors(cudaMemcpy(mat->outThresholds, h_thresholds, numThresholds*sizeof(float), cudaMemcpyHostToDevice));
	delete[] h_thresholds;

#ifdef BATCH_MODE
	checkCudaErrors(cudaMalloc(&mat->outThreshChanges, numThresholds*sizeof(float)));
	checkCudaErrors(cudaMemset(mat->outThreshChanges, 0, numThresholds*sizeof(float)));
#endif

	checkCudaErrors(cudaMalloc(&mat->outTDs, pars->numOutputNeurons*sizeof(float)));
	checkCudaErrors(cudaMalloc(&mat->inErrors, pars->numInputNeurons*sizeof(float)));
	checkCudaErrors(cudaMalloc(&mat->outErrors, pars->numOutputNeurons*sizeof(float)));

	mat->forwardSharedMem = getCalcFixedSharedSize(pars);
	mat->backwardSharedMem = getBPFixedNetSharedSize(pars);
	mat->numInputElements = pars->numInputNeurons;
	mat->numOutputElements = pars->numOutputNeurons;
}

void copyLayersToDevice(LayerCollection* layers) {
	for (size_t i = 0; i < layers->convMat.size(); i++) {
		ConvolutionMatrices* d_convMat;
		checkCudaErrors(cudaMalloc(&d_convMat, sizeof(ConvolutionMatrices)));
		checkCudaErrors(cudaMemcpy(d_convMat, &layers->convMat[i], sizeof(ConvolutionMatrices), cudaMemcpyHostToDevice));
		layers->d_convMat.push_back(d_convMat);
	}

	for (size_t i = 0; i < layers->convPars.size(); i++) {
		ConvolutionParameters* d_convPars;
		checkCudaErrors(cudaMalloc(&d_convPars, sizeof(ConvolutionParameters)));
		checkCudaErrors(cudaMemcpy(d_convPars, &layers->convPars[i], sizeof(ConvolutionParameters), cudaMemcpyHostToDevice));
		layers->d_convPars.push_back(d_convPars);
	}

	for (size_t i = 0; i < layers->mpMat.size(); i++) {
		MaxPoolMatrices* d_mpMat;
		checkCudaErrors(cudaMalloc(&d_mpMat, sizeof(MaxPoolMatrices)));
		checkCudaErrors(cudaMemcpy(d_mpMat, &layers->mpMat[i], sizeof(MaxPoolMatrices), cudaMemcpyHostToDevice));
		layers->d_mpMat.push_back(d_mpMat);
	}

	for (size_t i = 0; i < layers->mpPars.size(); i++) {
		MaxPoolParameters* d_mpPars;
		checkCudaErrors(cudaMalloc(&d_mpPars, sizeof(MaxPoolParameters)));
		checkCudaErrors(cudaMemcpy(d_mpPars, &layers->mpPars[i], sizeof(MaxPoolParameters), cudaMemcpyHostToDevice));
		layers->d_mpPars.push_back(d_mpPars);
	}

	for (size_t i = 0; i < layers->fixedMat.size(); i++) {
		FixedNetMatrices* d_fixedMat;
		checkCudaErrors(cudaMalloc(&d_fixedMat, sizeof(FixedNetMatrices)));
		checkCudaErrors(cudaMemcpy(d_fixedMat, &layers->fixedMat[i], sizeof(FixedNetMatrices), cudaMemcpyHostToDevice));
		layers->d_fixedMat.push_back(d_fixedMat);
	}

	for (size_t i = 0; i < layers->fixedPars.size(); i++) {
		FixedNetParameters* d_fixedPars;
		checkCudaErrors(cudaMalloc(&d_fixedPars, sizeof(FixedNetParameters)));
		checkCudaErrors(cudaMemcpy(d_fixedPars, &layers->fixedPars[i], sizeof(FixedNetParameters), cudaMemcpyHostToDevice));
		layers->d_fixedPars.push_back(d_fixedPars);
	}
}

LayerCollection createLayerCollection() {
	if (NUM_INPUTS != 64 && NUM_INPUTS != 136)
		throw std::runtime_error("Invalid number of inputs! Valid choices: 64, 136");

	LayerCollection layers;

	//set up layers manually for now
	if (NUM_INPUTS == 136) {
		ConvolutionParameters conv0;
		conv0.numInputLocs = 136;
		conv0.convSize = 8;
		conv0.numOutputLocs = 128;
		conv0.numInputNeurons = 1;
		conv0.numOutputNeurons = NUM_NEURONS;

		conv0.forBlockX = conv0.numInputNeurons;
		conv0.forBlockY = conv0.numOutputNeurons;
		conv0.forNBlockX = conv0.numOutputLocs;
		conv0.forNBlockY = 1;
		conv0.backBlockX = conv0.numOutputNeurons;
		conv0.backBlockY = 8;//conv0.numInputLocs/4;
		conv0.backNBlockX = conv0.numInputNeurons;
		conv0.backNBlockY = 1;

		layers.convPars.push_back(conv0);

		MaxPoolParameters mp0;
		mp0.numNeurons = NUM_NEURONS;
		mp0.numInputLocs = 128;
		mp0.numOutputLocs = 64;

		mp0.blockX = mp0.numNeurons;
		mp0.blockY = mp0.numOutputLocs / 2;
		mp0.nBlockX = 1;
		mp0.nBlockY = 2;

		layers.mpPars.push_back(mp0);
	}

	ConvolutionParameters conv1;
	conv1.numInputLocs = 64;
	conv1.convSize = 5;
	conv1.numOutputLocs = 60;
	if (NUM_INPUTS == 64)
		conv1.numInputNeurons = 1;
	else
		conv1.numInputNeurons = NUM_NEURONS;
	conv1.numOutputNeurons = NUM_NEURONS;

	conv1.forBlockX = conv1.numInputNeurons;
	conv1.forBlockY = conv1.numOutputNeurons;
	conv1.forNBlockX = conv1.numOutputLocs;
	conv1.forNBlockY = 1;
	conv1.backBlockX = conv1.numOutputNeurons;
	conv1.backBlockY = conv1.numInputLocs/4;
	conv1.backNBlockX = conv1.numInputNeurons;
	conv1.backNBlockY = 1;
	
	layers.convPars.push_back(conv1);

	MaxPoolParameters mp1;
	mp1.numNeurons = NUM_NEURONS;
	mp1.numInputLocs = 60;
	mp1.numOutputLocs = 30;
	
	mp1.blockX = mp1.numNeurons;
	mp1.blockY = mp1.numOutputLocs;
	mp1.nBlockX = 1;
	mp1.nBlockY = 1;

	layers.mpPars.push_back(mp1);

	ConvolutionParameters conv2;
	conv2.numInputLocs = 30;
	conv2.convSize = 5;
	conv2.numOutputLocs = 26;
	conv2.numInputNeurons = NUM_NEURONS;
	conv2.numOutputNeurons = NUM_NEURONS;

	conv2.forBlockX = conv2.numInputNeurons;
	conv2.forBlockY = conv2.numOutputNeurons;
	conv2.forNBlockX = conv2.numOutputLocs;
	conv2.forNBlockY = 1;
	conv2.backBlockX = conv2.numOutputNeurons;
	conv2.backBlockY = conv2.numInputLocs/2;
	conv2.backNBlockX = conv2.numInputNeurons;
	conv2.backNBlockY = 1;
	
	layers.convPars.push_back(conv2);

	MaxPoolParameters mp2;
	mp2.numNeurons = NUM_NEURONS;
	mp2.numInputLocs = 26;
	mp2.numOutputLocs = 13;
	
	mp2.blockX = mp2.numNeurons;
	mp2.blockY = mp2.numOutputLocs;
	mp2.nBlockX = 1;
	mp2.nBlockY = 1;

	layers.mpPars.push_back(mp2);

	ConvolutionParameters conv3;
	conv3.numInputLocs = 13;
	conv3.convSize = 4;
	conv3.numOutputLocs = 10;
	conv3.numInputNeurons = NUM_NEURONS;
	conv3.numOutputNeurons = NUM_NEURONS;

	conv3.forBlockX = conv3.numInputNeurons;
	conv3.forBlockY = conv3.numOutputNeurons;
	conv3.forNBlockX = conv3.numOutputLocs;
	conv3.forNBlockY = 1;
	conv3.backBlockX = conv3.numOutputNeurons;
	conv3.backBlockY = conv3.numInputLocs;
	conv3.backNBlockX = conv3.numInputNeurons;
	conv3.backNBlockY = 1;
	
	layers.convPars.push_back(conv3);

	MaxPoolParameters mp3;
	mp3.numNeurons = NUM_NEURONS;
	mp3.numInputLocs = 10;
	mp3.numOutputLocs = 5;
	
	mp3.blockX = mp3.numNeurons;
	mp3.blockY = mp3.numOutputLocs;
	mp3.nBlockX = 1;
	mp3.nBlockY = 1;

	layers.mpPars.push_back(mp3);

	ConvolutionParameters conv4;
	conv4.numInputLocs = 5;
	conv4.convSize = 2;
	conv4.numOutputLocs = 4;
	conv4.numInputNeurons = NUM_NEURONS;
	conv4.numOutputNeurons = NUM_NEURONS;

	conv4.forBlockX = conv4.numInputNeurons;
	conv4.forBlockY = conv4.numOutputNeurons;
	conv4.forNBlockX = conv4.numOutputLocs;
	conv4.forNBlockY = 1;
	conv4.backBlockX = conv4.numOutputNeurons;
	conv4.backBlockY = conv4.numInputLocs;
	conv4.backNBlockX = conv4.numInputNeurons;
	conv4.backNBlockY = 1;
	
	layers.convPars.push_back(conv4);

	MaxPoolParameters mp4;
	mp4.numNeurons = NUM_NEURONS;
	mp4.numInputLocs = 4;
	mp4.numOutputLocs = 2;
	
	mp4.blockX = mp4.numNeurons;
	mp4.blockY = mp4.numOutputLocs;
	mp4.nBlockX = 1;
	mp4.nBlockY = 1;

	layers.mpPars.push_back(mp4);

	FixedNetParameters fix1;
	fix1.numInputNeurons = 2 * NUM_NEURONS;
	fix1.numOutputNeurons = 2 * NUM_NEURONS;

	fix1.forBlockX = fix1.numInputNeurons;
	fix1.forBlockY = 1;
	fix1.forNBlockX = fix1.numOutputNeurons;
	fix1.forNBlockY = 1;
	fix1.backBlockX = fix1.numOutputNeurons;
	fix1.backBlockY = 1;
	fix1.backNBlockX = fix1.numInputNeurons;
	fix1.backNBlockY = 1;

	fix1.TFOutput = true;

	layers.fixedPars.push_back(fix1);

	FixedNetParameters fix2;
	fix2.numInputNeurons = 2 * NUM_NEURONS;
	fix2.numOutputNeurons = 1;

	fix2.forBlockX = fix2.numInputNeurons;
	fix2.forBlockY = 1;
	fix2.forNBlockX = fix2.numOutputNeurons;
	fix2.forNBlockY = 1;
	fix2.backBlockX = fix2.numOutputNeurons;
	fix2.backBlockY = 1;
	fix2.backNBlockX = fix2.numInputNeurons;
	fix2.backNBlockY = 1;

	fix2.TFOutput = false;

	layers.fixedPars.push_back(fix2);

	layers.numConvolutions = layers.convPars.size();
	layers.numFixedNets = layers.fixedPars.size();

	return layers;
}

void calculate(LayerCollection layers) {
	for (size_t j = 0; j < layers.numConvolutions; j++) {
		dim3 nBlocks(layers.convPars[j].forNBlockX, layers.convPars[j].forNBlockY);
		dim3 shape(layers.convPars[j].forBlockX, layers.convPars[j].forBlockY);
		convolve << <nBlocks, shape, layers.convMat[j].forwardSharedMem >> > (layers.d_convMat[j], layers.d_convPars[j]);
		checkCudaErrors(cudaPeekAtLastError());

		if (layers.mpMat.size() > j) {
			dim3 mpNBlocks(layers.mpPars[j].nBlockX, layers.mpPars[j].nBlockY);
			dim3 mpShape(layers.mpPars[j].blockX, layers.mpPars[j].blockY);
			calcMaxPool << <mpNBlocks, mpShape >> >(layers.d_mpMat[j], layers.d_mpPars[j]);
			checkCudaErrors(cudaPeekAtLastError());
		}
	}

	for (size_t j = 0; j < layers.numFixedNets; j++) {
		dim3 nBlocks(layers.fixedPars[j].forNBlockX, layers.fixedPars[j].forNBlockY);
		dim3 shape(layers.fixedPars[j].forBlockX, layers.fixedPars[j].forBlockY);
		calcFixedNet << <nBlocks, shape, layers.fixedMat[j].forwardSharedMem >> >(layers.d_fixedMat[j], layers.d_fixedPars[j]);
		checkCudaErrors(cudaPeekAtLastError());
	}
}

void backPropagate(LayerCollection layers) {
	for (size_t j = layers.numFixedNets - 1; j < layers.numFixedNets; j--) {
		dim3 nBlocks(layers.fixedPars[j].backNBlockX, layers.fixedPars[j].backNBlockY);
		dim3 shape(layers.fixedPars[j].backBlockX, layers.fixedPars[j].backBlockY);
		bpFixedNet << <nBlocks, shape, layers.fixedMat[j].backwardSharedMem >> >(layers.d_fixedMat[j], layers.d_fixedPars[j]);
		checkCudaErrors(cudaPeekAtLastError());
	}

	for (size_t j = layers.numConvolutions - 1; j < layers.numConvolutions; j--) {
		if (layers.mpMat.size() > j) {
			dim3 mpNBlocks(layers.mpPars[j].nBlockX, layers.mpPars[j].nBlockY);
			dim3 mpShape(layers.mpPars[j].blockX, layers.mpPars[j].blockY);
			bpMaxPool << <mpNBlocks, mpShape >> >(layers.d_mpMat[j], layers.d_mpPars[j]);
			checkCudaErrors(cudaPeekAtLastError());
		}

		dim3 nBlocks(layers.convPars[j].backNBlockX, layers.convPars[j].backNBlockY);
		dim3 shape(layers.convPars[j].backBlockX, layers.convPars[j].backBlockY);
		bpConvolution << <nBlocks, shape, layers.convMat[j].backwardSharedMem >> >(layers.d_convMat[j], layers.d_convPars[j]);
		checkCudaErrors(cudaPeekAtLastError());
	}
}

#ifdef BATCH_MODE
void batchUpdate(LayerCollection layers) {
	for (size_t i = 0; i < layers.numConvolutions; i++) {
		size_t numWeights = layers.convPars[i].numInputNeurons*layers.convPars[i].numOutputNeurons*layers.convPars[i].convSize;
		size_t numBlocks = numWeights % 256 == 0 ? numWeights / 256 : numWeights / 256 + 1;
		batchUpdateConvWeights << <numBlocks, 256 >> >(layers.d_convMat[i], layers.d_convPars[i]);
		checkCudaErrors(cudaPeekAtLastError());
	}

	for (size_t i = 0; i < layers.numFixedNets; i++) {
		size_t numWeights = layers.fixedPars[i].numInputNeurons*layers.fixedPars[i].numOutputNeurons;
		size_t numBlocks = numWeights % 256 == 0 ? numWeights / 256 : numWeights / 256 + 1;
		batchUpdateFixedWeights << <numBlocks, 256 >> >(layers.d_fixedMat[i], layers.d_fixedPars[i]);
		checkCudaErrors(cudaPeekAtLastError());
	}
}
#endif

float mean(std::vector<float> in) {
	float mean = 0;
	for (size_t i = 0; i < in.size(); i++) {
		mean += in[i];
	}
	mean /= in.size();
	return mean;
}

float stdev(std::vector<float> in, float mean) {
	float stdev = 0;
	for (size_t i = 0; i < in.size(); i++) {
		stdev += pow(in[i] - mean, 2);
	}
	stdev /= in.size();
	stdev = sqrt(stdev);
	return stdev;
}

float testSim(LayerCollection layers, std::string ofname) {
	float* d_inputs;
	if (layers.numConvolutions > 0) {
		if (layers.convPars[0].numInputLocs != NUM_INPUTS || layers.convPars[0].numInputNeurons != 1)
			throw std::runtime_error("inputs to first layer don't match data set");
		d_inputs = layers.convMat[0].inlayer;
	}
	else if (layers.numFixedNets > 0) {
		if (layers.fixedPars[0].numInputNeurons != NUM_INPUTS)
			throw std::runtime_error("inputs to first layer don't match data set");
		d_inputs = layers.fixedMat[0].inlayer;
	}
	else
		throw std::runtime_error("tried to run on a network with no convolutions and no fixed networks");

	float h_output[1];

	float error = 0;
	size_t numerrors = 0;
	size_t currentSample = 0;
	if (trainset.size() > 0)
		currentSample = trainset[0].samplenum;
	std::vector<float> sampleoutputs;

	std::ofstream outfile;
	if(ofname != "")
		outfile.open(ofname);

	for (size_t i = 0; i < trainset.size(); i++) {
		//----calculate----
		checkCudaErrors(cudaMemcpy(d_inputs, &trainset[i].inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice));

		calculate(layers);

		checkCudaErrors(cudaMemcpy(h_output, layers.fixedMat[layers.numFixedNets-1].outlayer, sizeof(float), cudaMemcpyDeviceToHost));
		//----end calculate-----

		sampleoutputs.push_back(h_output[0]);

		if (i == trainset.size() - 1 || trainset[i + 1].samplenum != currentSample) {
			numerrors++;
			float origmean = mean(sampleoutputs);
			float origdev = stdev(sampleoutputs, origmean);
			for (size_t j = 0; j < sampleoutputs.size(); j++) {
				if (fabs(sampleoutputs[j] - origmean) > origdev) {
					sampleoutputs.erase(sampleoutputs.begin() + j);
					j--;
				}
			}
			float newmean = mean(sampleoutputs);
			float newstdev = stdev(sampleoutputs, newmean);
			float correct = trainset[i].correctoutput;
			float newerror = newmean - correct;
			error += fabs(newerror);

			std::cout << "Sample " << currentSample << "| Actual: " << correct << " Measured: " << newmean << " +/- " << newstdev << " Error: " << newerror << std::endl;
			if (outfile.is_open()) {
				outfile << currentSample << " " << correct << " " << newmean << " " << newstdev << " " << newerror << std::endl;
			}

			sampleoutputs.clear();
			if (i < trainset.size() - 1)
				currentSample = trainset[i+1].samplenum;
		}
	}

	if (numerrors > 0) {
		error /= numerrors;
		//error = sqrt(error);
	}
	return error;
}

void randomizeTrainSet() {
	for (size_t i = 0; i < trainset.size(); i++) {
		size_t j = rand() % trainset.size();
		IOPair tmpio = trainset[i];
		trainset[i] = trainset[j];
		trainset[j] = tmpio;
	}
}

void saveExplicitTrainSet(std::string learnsetname) {
	std::stringstream learnsetss;
	learnsetss << datastring << learnsetname;
	std::ofstream outfile(learnsetss.str().c_str());

	for (size_t i = 0; i < trainset.size(); i++) {
		outfile << trainset[i].correctoutput*OUTPUT_DIVISOR << " | ";
		for (size_t j = 0; j < trainset[i].inputs.size(); j++) {
			outfile << trainset[i].inputs[j] << " ";
		}
		outfile << std::endl;
	}
}

size_t readExplicitTrainSet(std::string learnsetname, size_t begin, size_t numIOs) {
	trainset.clear();
	std::stringstream learnsetss;
	learnsetss << datastring << learnsetname;
	std::ifstream learnset(learnsetss.str().c_str());

	std::string line;
	std::string dum;
	size_t ionum = 0;
	while (getline(learnset, line)) {
		ionum++;
		if (numIOs > 0 && (ionum < begin || ionum >= begin + numIOs))
			continue;

		IOPair io;
		io.samplenum = 0;
		io.inputs.resize(NUM_INPUTS);
		std::stringstream lss(line);
		if (lss.eof()) throw std::runtime_error("Invalid train set file!");
		lss >> io.correctoutput;
		io.correctoutput /= OUTPUT_DIVISOR;
		lss >> dum;		//"|"
		for (size_t i = 0; i < NUM_INPUTS; i++) {
			if (lss.eof()) throw std::runtime_error("Invalid train set file!");
			lss >> io.inputs[i];
		}
		trainset.push_back(io);
	}
	return ionum;
}

void loadParameters(std::string parName) {
	std::ifstream infile(parName.c_str());
	std::string line;
	while (getline(infile, line)) {
		std::stringstream lss(line);
		std::string var;
		lss >> var;
		if (var == "NUM_INPUTS")
			lss >> NUM_INPUTS;
		else if (var == "INITIAL_OUTPUT_THRESHOLD")
			lss >> INITIAL_OUTPUT_THRESHOLD;
		else if (var == "OUTPUT_DIVISOR")
			lss >> OUTPUT_DIVISOR;
		else if (var == "savename")
			lss >> savename;
		else if (var == "trainstring")
			lss >> trainstring;
		else if (var == "randtrainstring")
			lss >> randtrainstring;
	}
}