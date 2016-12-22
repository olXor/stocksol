#include "stockrun.cuh"

std::vector<IOPair> trainset;
std::vector<IOPair> testset;

std::string datastring;
std::string savestring;

//parameters
size_t NUM_INPUTS = 64;
float INITIAL_OUTPUT_AVERAGE = 0.0f;
float OUTPUT_DIVISOR = 1.0f;
std::string savename = "weights";
std::string trainstring = "trainset";
std::string randtrainstring = "rantrainset";
bool tradeTypeLong = true;
size_t pairProximity = NUM_INPUTS*30;
size_t pairProximityMin = NUM_INPUTS;

bool convolutionsOn = true;
bool fixedNetOn = true;

bool binnedOutput = false;
float binWidth = 5.0f;
float binMin = 0.0f;
float binMax = 200.0f;
size_t numBins = 1;

float convDropoutWeight = 1.0f;
float savedConvDropoutWeight = 1.0f;	//for retrieving when dropout is enabled after being disabled
float fixedDropoutWeight = 1.0f;
float savedFixedDropoutWeight = 1.0f;

size_t numFixedHiddenNeurons = 2 * NUM_NEURONS;

bool sigmoidOnBinnedOutput = false;

float binPositiveOutput = BIN_POSITIVE_OUTPUT;
float binNegativeOutput = BIN_NEGATIVE_OUTPUT;

bool testSelectBinSum = false;
std::vector<float> testSelectBinMins;
std::vector<float> testSelectBinMaxes;
std::vector<float> oppositeSelectBinMins;
std::vector<float> oppositeSelectBinMaxes;

void setStrings(std::string data, std::string save) {
	datastring = data;
	savestring = save;
}

size_t readTrainSet(std::string learnsetname, size_t begin, size_t numIOs, bool overrideBinningSwitch, bool runOnTestSet) {
	std::vector<IOPair>* dataset;
	if (runOnTestSet)
		dataset = &testset;
	else
		dataset = &trainset;
	(*dataset).clear();
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
			if (tradeTypeLong)
				io.correctoutput = longprof / OUTPUT_DIVISOR;
			else
				io.correctoutput = shortprof / OUTPUT_DIVISOR;

			if (binnedOutput && !overrideBinningSwitch) {
				if (tradeTypeLong)
					io.correctbins = getBinnedOutput(longprof);
				else
					io.correctbins = getBinnedOutput(shortprof);
			}

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
			(*dataset).push_back(io);
		}
	}
	return ionum;
}

float runSim(LayerCollection layers, bool train, float customStepFactor, size_t samples, bool print, float* secondaryError, bool runOnTestSet) {
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

	float* h_output, *d_output;
	//checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
	checkCudaErrors(cudaHostAlloc(&h_output, numBins*sizeof(float), cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer(&d_output, h_output, 0));

	float stepfac = STEPFACTOR*customStepFactor/numBins;
	checkCudaErrors(cudaMemcpyAsync(layers.stepfactor, &stepfac, sizeof(float), cudaMemcpyHostToDevice));

	generateDropoutMask(&layers);

	cudaStream_t mainStream;
	//checkCudaErrors(cudaStreamCreate(&mainStream));
	mainStream = 0;

	cudaEvent_t calcDone;
	checkCudaErrors(cudaEventCreate(&calcDone));

	std::vector<IOPair>* dataset;
	if (runOnTestSet)
		dataset = &testset;
	else
		dataset = &trainset;

	size_t trainSamplesNum;
	if (samples > 0 && samples < dataset->size())
		trainSamplesNum = samples;
	else
		trainSamplesNum = dataset->size();
		
	float error = 0.0f;
	float secError = 0.0f;
	float secError2 = 0.0f;
	float secError3 = 0.0f;
	float secError4 = 0.0f;
	float secError5 = 0.0f;
	size_t numerrors = 0;
	
	//stuff associated with trying to select only the most certain winning trades
	std::vector<size_t> selectBinResults(numBins);
	for (size_t i = 0; i < numBins; i++)
		selectBinResults[i] = 0;
	float selectBinSum = 0.0f;
	size_t numSelected = 0;

	for (size_t i = 0; i < trainSamplesNum; i++) {
		numerrors++;

		//----calculate----
		checkCudaErrors(cudaMemcpyAsync(d_inputs, &(*dataset)[i].inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice, mainStream));

		calculate(layers, mainStream);

		//checkCudaErrors(cudaMemcpyAsync(h_output, layers.fixedMat[layers.numFixedNets-1].outlayer, sizeof(float), cudaMemcpyDeviceToHost, mainStream));

		//----end calculate-----

		if (binnedOutput)
			checkCudaErrors(cudaMemcpyAsync(layers.correctoutput, &(*dataset)[i].correctbins[0], numBins*sizeof(float), cudaMemcpyHostToDevice, mainStream));
		else
			checkCudaErrors(cudaMemcpyAsync(layers.correctoutput, &(*dataset)[i].correctoutput, sizeof(float), cudaMemcpyHostToDevice, mainStream));

		calculateOutputError << <1, numBins, 0, mainStream >> >(layers.d_fixedMat[layers.numFixedNets - 1], layers.stepfactor, layers.correctoutput, d_output);

		checkCudaErrors(cudaEventRecord(calcDone, mainStream));

		if (train) {
			backPropagate(layers, mainStream);
		}

		checkCudaErrors(cudaEventSynchronize(calcDone));

		float newerror = 0;
		bool select = false;
		if (binnedOutput) {
			size_t maxBin = 0;
			float maxCert = h_output[0];
			float avgPrediction = 0.0f;
			float totalCert = 0.0f;
			for (size_t j = 0; j < numBins; j++) {
				if (h_output[j] > maxCert) {
					maxBin = j;
					maxCert = h_output[j];
				}
				if (h_output[j] > 0.0f) {
					totalCert += h_output[j];
					avgPrediction += (binMin + binWidth*j + binWidth / 2)*h_output[j];
				}
				float unsquare = h_output[j] - (*dataset)[i].correctbins[j];
				newerror += unsquare*unsquare;
			}
			newerror = sqrt(newerror / numBins);
			error += newerror;
			avgPrediction /= totalCert;
			float unsquare = (avgPrediction - (*dataset)[i].correctoutput);
			secError += fabs(unsquare);
			secError2 += unsquare*unsquare;
			unsquare = (binMin + binWidth*maxBin + binWidth / 2 - (*dataset)[i].correctoutput);
			secError3 += fabs(unsquare);
			secError4 += unsquare*unsquare;

			if ((*dataset)[i].correctbins[maxBin] != binPositiveOutput){
				secError5 += 1.0f;
			}

			if (testSelectBinSum) {
				select = true;
				for (size_t j = 0; j < numBins; j++) {
					if (h_output[j] < testSelectBinMins[j] || h_output[j] > testSelectBinMaxes[j]) {
						select = false;
						break;
					}
				}
				if (select) {
					numSelected++;
					selectBinSum += (*dataset)[i].correctoutput;
					for (size_t j = 0; j < numBins; j++) {
						if ((*dataset)[i].correctbins[j] == BIN_POSITIVE_OUTPUT) {
							selectBinResults[j]++;
						}
					}
				}
			}
		}
		else {
			float unsquare = (h_output[0] - (*dataset)[i].correctoutput);
			newerror += unsquare*unsquare;
			error += sqrt(newerror);			//bit unnecessarily convoluted here perhaps
		}

		if (print) {
			/*
			std::cout << "Inputs: ";
			for (size_t j = 0; j < NUM_INPUTS; j++) {
				std::cout << dataset[i].inputs[j] << " ";
			}
			std::cout << std::endl;
			*/
			if (!binnedOutput)
				std::cout << "Output: " << h_output[0]*OUTPUT_DIVISOR << ", Correct: " << (*dataset)[i].correctoutput*OUTPUT_DIVISOR << ", Error: " << sqrt(newerror) << std::endl;
			else {
				std::cout << "Output: ";
				for (size_t j = 0; j < numBins; j++) {
					std::cout << h_output[j] << " ";
				}
				std::cout << "Correct: ";
				for (size_t j = 0; j < numBins; j++) {
					std::cout << (*dataset)[i].correctbins[j] << " ";
				}
				std::cout << "Error: " << newerror << " ";

				if (testSelectBinSum && select) {
					std::cout << " SELECTED (" << (*dataset)[i].correctoutput << ") ";
				}

				std::cout << std::endl;
			}
		}
	}

	if (numerrors > 0) {
		error /= numerrors;
		secError /= numerrors;
		secError2 /= numerrors;
		secError3 /= numerrors;
		secError4 /= numerrors;
		secError5 /= numerrors;
	}
	if (binnedOutput) {
		secError2 = sqrt(secError2);
		secError4 = sqrt(secError4);
	}
	if (secondaryError != NULL) {
		secondaryError[0] = secError;
		secondaryError[1] = secError2;
		secondaryError[2] = secError3;
		secondaryError[3] = secError4;
		secondaryError[4] = secError5;
	}

	if (testSelectBinSum) {
		std::cout << "Selected Sample Sum: " << selectBinSum << "/" << numSelected << "=" << selectBinSum / numSelected << std::endl;
		std::cout << "Selected Sample Bin Distribution: ";
		for (size_t i = 0; i < numBins; i++) {
			std::cout << selectBinResults[i] << "(" << 1.0f*selectBinResults[i] / numSelected << ") ";
		}
	}

	checkCudaErrors(cudaFreeHost(h_output));
	if (binnedOutput)
		return error;
	else
		return error*OUTPUT_DIVISOR;
}

float runPairedSim(PairedConvCollection layers, bool train, float customStepFactor, size_t samples, bool print, size_t pairsAveraged) {
	float* d_inputs1;
	float* d_inputs2;
	if (layers.conv1.numConvolutions > 0) {
		if (layers.conv1.convPars[0].numInputLocs != NUM_INPUTS || layers.conv1.convPars[0].numInputNeurons != 1)
			throw std::runtime_error("inputs to first layer don't match data set");
		d_inputs1 = layers.conv1.convMat[0].inlayer;
	}
	else if (layers.conv1.numFixedNets > 0) {
		if (layers.conv1.fixedPars[0].numInputNeurons != NUM_INPUTS)
			throw std::runtime_error("inputs to first layer don't match data set");
		d_inputs1 = layers.conv1.fixedMat[0].inlayer;
	}
	else
		throw std::runtime_error("tried to run on a network with no convolutions and no fixed networks");

	if (layers.conv2.numConvolutions > 0) {
		if (layers.conv2.convPars[0].numInputLocs != NUM_INPUTS || layers.conv2.convPars[0].numInputNeurons != 1)
			throw std::runtime_error("inputs to first layer don't match data set");
		d_inputs2 = layers.conv2.convMat[0].inlayer;
	}
	else if (layers.conv2.numFixedNets > 0) {
		if (layers.conv2.fixedPars[0].numInputNeurons != NUM_INPUTS)
			throw std::runtime_error("inputs to first layer don't match data set");
		d_inputs2 = layers.conv2.fixedMat[0].inlayer;
	}
	else
		throw std::runtime_error("tried to run on a network with no convolutions and no fixed networks");

	float* h_output, *d_output;
	//checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
	checkCudaErrors(cudaHostAlloc(&h_output, numBins*sizeof(float), cudaHostAllocMapped));
	checkCudaErrors(cudaHostGetDevicePointer(&d_output, h_output, 0));

	float stepfac = STEPFACTOR*customStepFactor;
	checkCudaErrors(cudaMemcpyAsync(layers.fixed.stepfactor, &stepfac, sizeof(float), cudaMemcpyHostToDevice));

	cudaStream_t mainStream;
	//checkCudaErrors(cudaStreamCreate(&mainStream));
	mainStream = 0;

	cudaEvent_t calcDone;
	checkCudaErrors(cudaEventCreate(&calcDone));

	size_t trainSamplesNum;
	if (samples > 0 && samples < trainset.size())
		trainSamplesNum = samples;
	else
		trainSamplesNum = trainset.size();
		
	float error = 0;
	size_t numerrors = 0;
	int proxRange = max((int)pairProximity - (int)pairProximityMin, 1);
	for (size_t i = 0; i < trainSamplesNum; i++) {
		size_t runsLeft;
		if (pairsAveraged > 0)
			runsLeft = pairsAveraged;
		else
			runsLeft = 1;

		float avgoutput = 0;
		size_t numOutputAvgs = 0;
		for (; runsLeft > 0; runsLeft--) {
			size_t j = i - (rand() % (proxRange)) - pairProximityMin;
			if (j >= trainSamplesNum)
				continue;

			//----calculate----
			checkCudaErrors(cudaMemcpyAsync(d_inputs1, &trainset[j].inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice, mainStream));
			checkCudaErrors(cudaMemcpyAsync(d_inputs2, &trainset[i].inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice, mainStream));

			calculate(layers.conv1, mainStream);
			calculate(layers.conv2, mainStream);
			calculate(layers.fixed, mainStream);

			//----end calculate-----

			float correctdifference = trainset[i].correctoutput - trainset[j].correctoutput;

			checkCudaErrors(cudaMemcpyAsync(layers.fixed.correctoutput, &correctdifference, sizeof(float), cudaMemcpyHostToDevice, mainStream));

			calculateOutputError << <1, numBins, 0, mainStream >> >(layers.fixed.d_fixedMat[layers.fixed.numFixedNets - 1], layers.fixed.stepfactor, layers.fixed.correctoutput, d_output);

			checkCudaErrors(cudaEventRecord(calcDone, mainStream));

			if (train) {
				backPropagate(layers.fixed, mainStream);
				backPropagate(layers.conv1, mainStream);
				backPropagate(layers.conv2, mainStream);
			}

			checkCudaErrors(cudaEventSynchronize(calcDone));

			if (pairsAveraged == 0) {
				float newerror = (h_output[0] - correctdifference);
				//error += newerror*newerror;
				error += fabs(newerror);
				numerrors++;
				
				if (print) {
					/*
					std::cout << "Inputs: ";
					for (size_t j = 0; j < NUM_INPUTS; j++) {
					std::cout << trainset[i].inputs[j] << " ";
					}
					std::cout << std::endl;
					*/
					std::cout << "Correct profits: " << trainset[j].correctoutput*OUTPUT_DIVISOR << ", " << trainset[i].correctoutput*OUTPUT_DIVISOR << " Difference: " << correctdifference*OUTPUT_DIVISOR << ", Output: " << h_output[0] * OUTPUT_DIVISOR << ", Error: " << newerror*OUTPUT_DIVISOR << std::endl;
				}
			}
			else {
				float newoutput = trainset[j].correctoutput + h_output[0];
				avgoutput += newoutput;
				numOutputAvgs++;
			}
		}

		if (pairsAveraged > 0) {
			if (numOutputAvgs > 0)
				avgoutput /= numOutputAvgs;
			else
				continue;

			float newerror = (avgoutput - trainset[i].correctoutput);
			error += fabs(newerror);
			numerrors++;
			if (print)
				std::cout << "Correct profit: " << trainset[i].correctoutput*OUTPUT_DIVISOR << ", Output: " << avgoutput*OUTPUT_DIVISOR << ", Error: " << newerror*OUTPUT_DIVISOR << std::endl;
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

bool loadWeights(LayerCollection layers, std::string fname) {
	std::stringstream fss;
	fss << savestring << fname;

	if (!PathFileExists(fss.str().c_str())) {
		std::cout << "No weights file found; starting with random weights" << std::endl;
		return false;
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

	return true;
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
		else if (i == 0 && layers->numConvolutions != 0){
			size_t lastConv = layers->numConvolutions - 1;
			numPrevOutputs = layers->mpMat[lastConv].numOutputElements;
			prevOutLayer = layers->mpMat[lastConv].outlayer;
			prevOutErrors = layers->mpMat[lastConv].outError;
		}
		else if (i != 0) {
			numPrevOutputs = layers->fixedMat[i - 1].numOutputElements;
			prevOutLayer = layers->fixedMat[i - 1].outlayer;
			prevOutErrors = layers->fixedMat[i - 1].outErrors;
		}
		else
			continue;

		if (numPrevOutputs != layers->fixedMat[i].numInputElements)
			throwLayerLinkError();

		FixedNetMatrices* mat = &layers->fixedMat[i];
		checkCudaErrors(cudaFree(mat->inlayer));
		mat->inlayer = prevOutLayer;
		checkCudaErrors(cudaFree(mat->inErrors));
		mat->inErrors = prevOutErrors;
	}

	copyLayersToDevice(layers);

	checkCudaErrors(cudaMalloc(&layers->stepfactor, sizeof(float)));
	checkCudaErrors(cudaMalloc(&layers->correctoutput, numBins*sizeof(float)));

	initializeDropoutRNG(layers);
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

	size_t numDropouts = pars->numOutputNeurons*pars->numOutputLocs;
	float* h_dropouts = new float[numDropouts];
	for (size_t i = 0; i < numDropouts; i++) {
		h_dropouts[i] = 1.0f;
	}
	checkCudaErrors(cudaMalloc(&mat->dropoutFactors, numDropouts*sizeof(float)));
	checkCudaErrors(cudaMemcpy(mat->dropoutFactors, h_dropouts, numDropouts*sizeof(float), cudaMemcpyHostToDevice));
	delete h_dropouts;

	checkCudaErrors(cudaMalloc(&mat->randStates, numDropouts*sizeof(curandState)));

	mat->forwardSharedMem = getConvolveSharedSize(pars);
	mat->backPropErrSharedMem = getBPEConvolutionSharedSize(pars);
	mat->backUpdateSharedMem = getBackUpdateConvolutionSharedSize(pars);
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

	if (last && !binnedOutput)
		h_thresholds[0] = -INITIAL_OUTPUT_AVERAGE/OUTPUT_DIVISOR;
	if (last && binnedOutput) {
		for (size_t i = 0; i < numThresholds; i++) {
			if (sigmoidOnBinnedOutput)
				h_thresholds[i] = 0.0f;
			else
				h_thresholds[i] = -binNegativeOutput - (binPositiveOutput - binNegativeOutput) / 2;
		}
	}

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

	size_t numDropouts = pars->numOutputNeurons;
	float* h_dropouts = new float[numDropouts];
	for (size_t i = 0; i < numDropouts; i++) {
		h_dropouts[i] = 1.0f;
	}
	checkCudaErrors(cudaMalloc(&mat->dropoutFactors, numDropouts*sizeof(float)));
	checkCudaErrors(cudaMemcpy(mat->dropoutFactors, h_dropouts, numDropouts*sizeof(float), cudaMemcpyHostToDevice));
	delete h_dropouts;

	checkCudaErrors(cudaMalloc(&mat->randStates, numDropouts*sizeof(curandState)));

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

void freeDeviceLayers(LayerCollection* layers) {
	for (size_t i = 0; i < layers->d_convMat.size(); i++) {
		checkCudaErrors(cudaFree(layers->d_convMat[i]));
	}
	layers->d_convMat.resize(0);

	for (size_t i = 0; i < layers->d_convPars.size(); i++) {
		checkCudaErrors(cudaFree(layers->d_convPars[i]));
	}
	layers->d_convPars.resize(0);

	for (size_t i = 0; i < layers->d_mpMat.size(); i++) {
		checkCudaErrors(cudaFree(layers->d_mpMat[i]));
	}
	layers->d_mpMat.resize(0);

	for (size_t i = 0; i < layers->d_mpPars.size(); i++) {
		checkCudaErrors(cudaFree(layers->d_mpPars[i]));
	}
	layers->d_mpPars.resize(0);

	for (size_t i = 0; i < layers->d_fixedMat.size(); i++) {
		checkCudaErrors(cudaFree(layers->d_fixedMat[i]));
	}
	layers->d_fixedMat.resize(0);

	for (size_t i = 0; i < layers->d_fixedPars.size(); i++) {
		checkCudaErrors(cudaFree(layers->d_fixedPars[i]));
	}
	layers->d_fixedPars.resize(0);
}

LayerCollection createLayerCollection(size_t numInputs, int LCType) {
	if (numInputs == 0)
		numInputs = NUM_INPUTS;

	if ((LCType == FULL_NETWORK || LCType == CONV_ONLY) && numInputs != 64 && numInputs != 136)
		throw std::runtime_error("Invalid number of inputs! Valid choices: 64, 136");

	LayerCollection layers;

	//set up layers manually for now

	if (LCType == FULL_NETWORK || LCType == CONV_ONLY) {
		if (numInputs == 136) {
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
			conv0.backPropErrBlockX = conv0.numOutputNeurons;
			conv0.backPropErrBlockY = conv0.convSize;
			conv0.backPropErrNBlockX = conv0.numInputNeurons;
			conv0.backPropErrNBlockY = conv0.numInputLocs;
			conv0.backUpdateBlockX = conv0.numOutputLocs;
			conv0.backUpdateNBlockX = conv0.numInputNeurons;
			conv0.backUpdateNBlockY = conv0.numOutputNeurons;
			conv0.backUpdateNBlockZ = conv0.convSize;

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
		if (numInputs == 64)
			conv1.numInputNeurons = 1;
		else
			conv1.numInputNeurons = NUM_NEURONS;
		conv1.numOutputNeurons = NUM_NEURONS;

		conv1.forBlockX = conv1.numInputNeurons;
		conv1.forBlockY = conv1.numOutputNeurons;
		conv1.forNBlockX = conv1.numOutputLocs;
		conv1.forNBlockY = 1;
		conv1.backPropErrBlockX = conv1.numOutputNeurons;
		conv1.backPropErrBlockY = conv1.convSize;
		conv1.backPropErrNBlockX = conv1.numInputNeurons;
		conv1.backPropErrNBlockY = conv1.numInputLocs;
		conv1.backUpdateBlockX = conv1.numOutputLocs;
		conv1.backUpdateNBlockX = conv1.numInputNeurons;
		conv1.backUpdateNBlockY = conv1.numOutputNeurons;
		conv1.backUpdateNBlockZ = conv1.convSize;

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
		conv2.backPropErrBlockX = conv2.numOutputNeurons;
		conv2.backPropErrBlockY = conv2.convSize;
		conv2.backPropErrNBlockX = conv2.numInputNeurons;
		conv2.backPropErrNBlockY = conv2.numInputLocs;
		conv2.backUpdateBlockX = conv2.numOutputLocs;
		conv2.backUpdateNBlockX = conv2.numInputNeurons;
		conv2.backUpdateNBlockY = conv2.numOutputNeurons;
		conv2.backUpdateNBlockZ = conv2.convSize;

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
		conv3.backPropErrBlockX = conv3.numOutputNeurons;
		conv3.backPropErrBlockY = conv3.convSize;
		conv3.backPropErrNBlockX = conv3.numInputNeurons;
		conv3.backPropErrNBlockY = conv3.numInputLocs;
		conv3.backUpdateBlockX = conv3.numOutputLocs;
		conv3.backUpdateNBlockX = conv3.numInputNeurons;
		conv3.backUpdateNBlockY = conv3.numOutputNeurons;
		conv3.backUpdateNBlockZ = conv3.convSize;

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
		conv4.backPropErrBlockX = conv4.numOutputNeurons;
		conv4.backPropErrBlockY = conv4.convSize;
		conv4.backPropErrNBlockX = conv4.numInputNeurons;
		conv4.backPropErrNBlockY = conv4.numInputLocs;
		conv4.backUpdateBlockX = conv4.numOutputLocs;
		conv4.backUpdateNBlockX = conv4.numInputNeurons;
		conv4.backUpdateNBlockY = conv4.numOutputNeurons;
		conv4.backUpdateNBlockZ = conv4.convSize;

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
	}

	if (LCType == FULL_NETWORK) {
		FixedNetParameters fix1;
		fix1.numInputNeurons = 2 * NUM_NEURONS;
		fix1.numOutputNeurons = numFixedHiddenNeurons;

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
		fix2.numInputNeurons = numFixedHiddenNeurons;
		fix2.numOutputNeurons = numBins;

		fix2.forBlockX = fix2.numInputNeurons;
		fix2.forBlockY = 1;
		fix2.forNBlockX = fix2.numOutputNeurons;
		fix2.forNBlockY = 1;
		fix2.backBlockX = fix2.numOutputNeurons;
		fix2.backBlockY = 1;
		fix2.backNBlockX = fix2.numInputNeurons;
		fix2.backNBlockY = 1;

		if (binnedOutput && sigmoidOnBinnedOutput) {
			fix2.transferType = TRANSFER_TYPE_SIGMOID;
			fix2.TFOutput = true;
		}
		else
			fix2.TFOutput = false;

		layers.fixedPars.push_back(fix2);
	}
	else if (LCType == FIXED_ONLY) {
		FixedNetParameters fix1;
		fix1.numInputNeurons = numInputs;
		fix1.numOutputNeurons = numFixedHiddenNeurons;

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
		fix2.numInputNeurons = numFixedHiddenNeurons;
		fix2.numOutputNeurons = numBins;

		fix2.forBlockX = fix2.numInputNeurons;
		fix2.forBlockY = 1;
		fix2.forNBlockX = fix2.numOutputNeurons;
		fix2.forNBlockY = 1;
		fix2.backBlockX = fix2.numOutputNeurons;
		fix2.backBlockY = 1;
		fix2.backNBlockX = fix2.numInputNeurons;
		fix2.backNBlockY = 1;

		if (binnedOutput && sigmoidOnBinnedOutput) {
			fix2.transferType = TRANSFER_TYPE_SIGMOID;
			fix2.TFOutput = true;
		}
		else
			fix2.TFOutput = false;

		layers.fixedPars.push_back(fix2);
	}

	layers.numConvolutions = layers.convPars.size();
	layers.numFixedNets = layers.fixedPars.size();

	return layers;
}

void calculate(LayerCollection layers, cudaStream_t stream) {
	for (size_t j = 0; j < layers.numConvolutions; j++) {
		dim3 nBlocks(layers.convPars[j].forNBlockX, layers.convPars[j].forNBlockY);
		dim3 shape(layers.convPars[j].forBlockX, layers.convPars[j].forBlockY);
		convolve << <nBlocks, shape, layers.convMat[j].forwardSharedMem, stream >> > (layers.d_convMat[j], layers.d_convPars[j]);
		checkCudaErrors(cudaPeekAtLastError());

		if (layers.mpMat.size() > j) {
			dim3 mpNBlocks(layers.mpPars[j].nBlockX, layers.mpPars[j].nBlockY);
			dim3 mpShape(layers.mpPars[j].blockX, layers.mpPars[j].blockY);
			calcMaxPool << <mpNBlocks, mpShape, 0, stream >> >(layers.d_mpMat[j], layers.d_mpPars[j]);
			checkCudaErrors(cudaPeekAtLastError());
		}
	}

	for (size_t j = 0; j < layers.numFixedNets; j++) {
		dim3 nBlocks(layers.fixedPars[j].forNBlockX, layers.fixedPars[j].forNBlockY);
		dim3 shape(layers.fixedPars[j].forBlockX, layers.fixedPars[j].forBlockY);
		calcFixedNet << <nBlocks, shape, layers.fixedMat[j].forwardSharedMem, stream >> >(layers.d_fixedMat[j], layers.d_fixedPars[j]);
		checkCudaErrors(cudaPeekAtLastError());
	}
}

void backPropagate(LayerCollection layers, cudaStream_t stream) {
	for (size_t j = layers.numFixedNets - 1; j < layers.numFixedNets; j--) {
		dim3 nBlocks(layers.fixedPars[j].backNBlockX, layers.fixedPars[j].backNBlockY);
		dim3 shape(layers.fixedPars[j].backBlockX, layers.fixedPars[j].backBlockY);
		bpFixedNet << <nBlocks, shape, layers.fixedMat[j].backwardSharedMem, stream >> >(layers.d_fixedMat[j], layers.d_fixedPars[j]);
		checkCudaErrors(cudaPeekAtLastError());
	}

	for (size_t j = layers.numConvolutions - 1; j < layers.numConvolutions; j--) {
		if (layers.mpMat.size() > j) {
			dim3 mpNBlocks(layers.mpPars[j].nBlockX, layers.mpPars[j].nBlockY);
			dim3 mpShape(layers.mpPars[j].blockX, layers.mpPars[j].blockY);
			bpMaxPool << <mpNBlocks, mpShape, 0, stream >> >(layers.d_mpMat[j], layers.d_mpPars[j]);
			checkCudaErrors(cudaPeekAtLastError());
		}

		dim3 nBlocks(layers.convPars[j].backPropErrNBlockX, layers.convPars[j].backPropErrNBlockY);
		dim3 shape(layers.convPars[j].backPropErrBlockX, layers.convPars[j].backPropErrBlockY);
		propagateErrorConvolution << <nBlocks, shape, layers.convMat[j].backPropErrSharedMem, stream >> >(layers.d_convMat[j], layers.d_convPars[j]);
		checkCudaErrors(cudaPeekAtLastError());

		dim3 nBlocks2(layers.convPars[j].backUpdateNBlockX, layers.convPars[j].backUpdateNBlockY, layers.convPars[j].backUpdateNBlockZ);
		dim3 shape2(layers.convPars[j].backUpdateBlockX);
		updateWeightsConvolution << <nBlocks2, shape2, layers.convMat[j].backUpdateSharedMem, stream >> >(layers.d_convMat[j], layers.d_convPars[j]);
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

	float* h_output = new float[numBins];

	float error = 0;
	size_t numerrors = 0;
	size_t currentSample = 0;
	if (trainset.size() > 0)
		currentSample = trainset[0].samplenum;
	std::vector<float> sampleoutputs;
	float* samplebins = new float[numBins];
	for (size_t i = 0; i < numBins; i++)
		samplebins[i] = 0.0f;

	std::ofstream outfile;
	if(ofname != "")
		outfile.open(ofname);

	for (size_t i = 0; i < trainset.size(); i++) {
		//----calculate----
		checkCudaErrors(cudaMemcpy(d_inputs, &trainset[i].inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice));

		calculate(layers);

		checkCudaErrors(cudaMemcpy(h_output, layers.fixedMat[layers.numFixedNets - 1].outlayer, numBins*sizeof(float), cudaMemcpyDeviceToHost));
		//----end calculate-----

		if (!binnedOutput) {
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
					currentSample = trainset[i + 1].samplenum;
			}
		}
		else {
			for (size_t j = 0; j < numBins; j++) {
				samplebins[j] += h_output[j];
			}

			if (i == trainset.size() - 1 || trainset[i + 1].samplenum != currentSample) {
				numerrors++;

				float maxProb = 0.0f;
				float totalProb = 0.0f;
				size_t bestBin = 0;
				size_t correctBin = 0;

				for (size_t j = 0; j < numBins; j++) {
					if (samplebins[j] > maxProb) {
						maxProb = samplebins[j];
						bestBin = j;
					}
					if (samplebins[j] > 0.0f)
						totalProb += samplebins[j];
					if (trainset[i].correctbins[j] == binPositiveOutput) {
						correctBin = j;
					}
				}

				std::cout << "Sample " << currentSample << "| Actual: " << binMin + correctBin*binWidth << "-" << binMin + (correctBin + 1)*binWidth << " Measured: " << binMin + bestBin*binWidth << "-" << binMin + (bestBin + 1)*binWidth << " with confidence " << maxProb / totalProb << std::endl;
				if (outfile.is_open()) {
					outfile << currentSample << " " << binMin + correctBin*binWidth << " " << binMin + bestBin*binWidth << " " << maxProb / totalProb << std::endl;
				}

				for (size_t j = 0; j < numBins; j++) {
					samplebins[j] = 0.0f;
				}
				if (i < trainset.size() - 1)
					currentSample = trainset[i + 1].samplenum;
			}
		}
	}

	delete[] h_output;
	delete[] samplebins;

	if (numerrors > 0) {
		error /= numerrors;
		//error = sqrt(error);
	}
	return error;
}

float sampleTestSim(LayerCollection layers, std::ofstream* outfile, bool testPrintSampleAll) {
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

	float* h_output = new float[numBins];

	float error = 0;
	size_t numerrors = 0;
	size_t currentSample = 0;
	if (trainset.size() > 0)
		currentSample = trainset[0].samplenum;
	std::vector<float> sampleoutputs;
	float* samplebins = new float[numBins];
	for (size_t i = 0; i < numBins; i++)
		samplebins[i] = 0.0f;
	size_t numCorrectlyBinned = 0;
	size_t numIncorrectlyBinned = 0;
	float avgMAV = 0.0f;    //mean absolute value
	float avgRMS = 0.0f;
	float binMAV = 0.0f;	
	float binRMS = 0.0f;
	float squareMAV = 0.0f;
	float squareRMS = 0.0f;
	size_t avgClose = 0;
	size_t binClose = 0;
	size_t squareClose = 0;

	disableDropout();
	generateDropoutMask(&layers);

	for (size_t i = 0; i < trainset.size(); i++) {
		//----calculate----
		checkCudaErrors(cudaMemcpy(d_inputs, &trainset[i].inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice));

		calculate(layers);

		checkCudaErrors(cudaMemcpy(h_output, layers.fixedMat[layers.numFixedNets - 1].outlayer, numBins*sizeof(float), cudaMemcpyDeviceToHost));
		//----end calculate-----

		if (binnedOutput) {
			for (size_t j = 0; j < numBins; j++)
				samplebins[j] += h_output[j];
		}
		else {
			sampleoutputs.push_back(h_output[0]);
		}

		if (i == trainset.size() - 1 || trainset[i + 1].samplenum != currentSample) {
			numerrors++;
			if (binnedOutput) {
				float maxProb = 0.0f;
				float totalProb = 0.0f;
				float totalSquare = 0.0f;
				size_t bestBin = 0;
				size_t correctBin = 0;
				float correctoutput = trainset[i].correctoutput;
				float avgRes = 0.0f;
				float squareRes = 0.0f;

				for (size_t j = 0; j < numBins; j++) {
					if (samplebins[j] > maxProb) {
						maxProb = samplebins[j];
						bestBin = j;
					}
					if (samplebins[j] > 0.0f) {
						totalProb += samplebins[j];
						totalSquare += samplebins[j] * samplebins[j];
						avgRes += samplebins[j]*(binMin + j*binWidth + binWidth / 2);
						squareRes += samplebins[j] * samplebins[j] * (binMin + j*binWidth + binWidth / 2);
					}
					if (trainset[i].correctbins[j] == binPositiveOutput) {
						correctBin = j;
					}
				}

				avgRes /= totalProb;
				squareRes /= totalSquare;
				float binRes = binMin + bestBin*binWidth + binWidth / 2;
				avgMAV += fabs(avgRes - correctoutput);
				avgRMS += (avgRes - correctoutput)*(avgRes - correctoutput);
				binMAV += fabs(binRes - correctoutput);
				binRMS += (binRes - correctoutput)*(binRes - correctoutput);
				squareMAV += fabs(squareRes - correctoutput);
				squareRMS += (squareRes - correctoutput)*(squareRes - correctoutput);

				if (fabs(avgRes - correctoutput) <= binWidth)
					avgClose++;
				if (fabs(binRes - correctoutput) <= binWidth)
					binClose++;
				if (fabs(squareRes - correctoutput) <= binWidth)
					squareClose++;
				std::cout << "Sample " << currentSample << "| Actual: " << correctoutput << " Top Bin: " << binMin + bestBin*binWidth << "-" << binMin + (bestBin + 1)*binWidth << " with confidence " << maxProb / totalProb << " (E: " << binRes - correctoutput << ") Weighted Average: " << avgRes << " (E: " << avgRes - correctoutput << ") Squared Average: " << squareRes << " (E: " << squareRes - correctoutput << ")";
				if (correctBin == bestBin) {
					std::cout << " CORRECT" << std::endl;
					numCorrectlyBinned++;
				}
				else {
					std::cout << " INCORRECT" << std::endl;
					numIncorrectlyBinned++;
					if (correctBin > bestBin)
						error += binWidth*(correctBin - bestBin);
					else
						error += binWidth*(bestBin - correctBin);
				}
				if (outfile->is_open()) {
					(*outfile) << currentSample << " " << correctoutput << " " << binRes << " " << avgRes << " " << squareRes << std::endl;
				}

				for (size_t j = 0; j < numBins; j++) {
					samplebins[j] = 0.0f;
				}
				if (i < trainset.size() - 1)
					currentSample = trainset[i + 1].samplenum;
			}
			else {
				float origmean = mean(sampleoutputs);
				float origdev = stdev(sampleoutputs, origmean);
				for (size_t j = 0; j < sampleoutputs.size(); j++) {
					if (testPrintSampleAll) {
						std::cout << "   Sample " << currentSample << "| Actual: " << trainset[i].correctoutput << " Measured: " << sampleoutputs[j] << " Error: " << sampleoutputs[j] - trainset[i].correctoutput;
						(*outfile) << "   Sample " << currentSample << "| Actual: " << trainset[i].correctoutput << " Measured: " << sampleoutputs[j] << " Error: " << sampleoutputs[j] - trainset[i].correctoutput;
					}
					if (fabs(sampleoutputs[j] - origmean) > origdev) {
						sampleoutputs.erase(sampleoutputs.begin() + j);
						j--;
						if (testPrintSampleAll) {
							std::cout << " DISCARDED";
							(*outfile) << " DISCARDED";
						}
					}
					if (testPrintSampleAll) {
						std::cout << std::endl;
						(*outfile) << std::endl;
					}
				}
				float newmean = mean(sampleoutputs);
				float newstdev = stdev(sampleoutputs, newmean);
				float correct = trainset[i].correctoutput;
				float newerror = newmean - correct;
				error += fabs(newerror);

				std::cout << "Sample " << currentSample << "| Actual: " << correct << " Measured: " << newmean << " +/- " << newstdev << " Error: " << newerror << std::endl;
				if ((*outfile).is_open()) {
					(*outfile) << currentSample << " " << correct << " " << newmean << " " << newstdev << " " << newerror << std::endl;
				}

				sampleoutputs.clear();
			}
			if (i < trainset.size() - 1)
				currentSample = trainset[i + 1].samplenum;
		}
	}

	delete[] h_output;
	delete[] samplebins;

	if (numerrors > 0) {
		if (binnedOutput) {
			std::cout << "Correctly Binned: " << numCorrectlyBinned << "/" << numCorrectlyBinned + numIncorrectlyBinned << "(" << 1.0f*numCorrectlyBinned / (numCorrectlyBinned + numIncorrectlyBinned) << ")" << std::endl;
			avgMAV /= numerrors;
			binMAV /= numerrors;
			squareMAV /= numerrors;
			avgRMS = sqrt(avgRMS / numerrors);
			binRMS = sqrt(binRMS / numerrors);
			squareRMS = sqrt(squareRMS / numerrors);

			std::cout << "Avg MAV: " << avgMAV << " Avg RMS: " << avgRMS << " Bin MAV: " << binMAV << " Bin RMS: " << binRMS << " Square MAV: " << squareMAV << " Square RMS: " << squareRMS << std::endl;
			std::cout << "Within Bin Width: " << "Avg: " << avgClose << "/" << numerrors << "(" << 1.0f*avgClose / numerrors << ")" << "Bin: " << binClose << "/" << numerrors << "(" << 1.0f*binClose / numerrors << ")" << "Squared: " << squareClose << "/" << numerrors << "(" << 1.0f*squareClose / numerrors << ")" << std::endl;
		}

		error /= numerrors;
		//error = sqrt(error);
	}
	return error;
}

void randomizeTrainSet(size_t maxIndex) {
	if (maxIndex == 0 || maxIndex > trainset.size())
		maxIndex = trainset.size();
	for (size_t i = 0; i < maxIndex; i++) {
		size_t j = rand() % maxIndex;
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

size_t readExplicitTrainSet(std::string learnsetname, size_t begin, size_t numIOs, bool runOnTestSet) {
	std::vector<IOPair>*dataset;
	if (runOnTestSet)
		dataset = &testset;
	else
		dataset = &trainset;

	(*dataset).clear();
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

		float correctoutput;
		lss >> correctoutput;
		io.correctoutput = correctoutput / OUTPUT_DIVISOR;

		if (binnedOutput) {
			io.correctbins = getBinnedOutput(correctoutput);
		}

		lss >> dum;		//"|"
		for (size_t i = 0; i < NUM_INPUTS; i++) {
			if (lss.eof()) throw std::runtime_error("Invalid train set file!");
			lss >> io.inputs[i];
		}
		(*dataset).push_back(io);
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
		else if (var == "INITIAL_OUTPUT_AVERAGE")
			lss >> INITIAL_OUTPUT_AVERAGE;
		else if (var == "OUTPUT_DIVISOR")
			lss >> OUTPUT_DIVISOR;
		else if (var == "savename")
			lss >> savename;
		else if (var == "trainstring")
			lss >> trainstring;
		else if (var == "randtrainstring")
			lss >> randtrainstring;
		else if (var == "tradeTypeLong")
			lss >> tradeTypeLong;
		else if (var == "pairProximity")
			lss >> pairProximity;
		else if (var == "pairProximityMin")
			lss >> pairProximityMin;
		else if (var == "convolutionsOn")
			lss >> convolutionsOn;
		else if (var == "fixedNetOn")
			lss >> fixedNetOn;
		else if (var == "binnedOutput")
			lss >> binnedOutput;
		else if (var == "binMin")
			lss >> binMin;
		else if (var == "binMax")
			lss >> binMax;
		else if (var == "binWidth")
			lss >> binWidth;
		else if (var == "convDropoutWeight") {
			lss >> convDropoutWeight;
			savedConvDropoutWeight = convDropoutWeight;
		}
		else if (var == "fixedDropoutWeight") {
			lss >> fixedDropoutWeight;
			savedFixedDropoutWeight = fixedDropoutWeight;
		}
		else if (var == "numFixedHiddenNeurons")
			lss >> numFixedHiddenNeurons;
		else if (var == "sigmoidOnBinnedOutput") {
			lss >> sigmoidOnBinnedOutput;
			if (sigmoidOnBinnedOutput) {
				binPositiveOutput = 1.0f;
				binNegativeOutput = 0.0f;
			}
		}
		else if (var == "testSelectBinSum")
			lss >> testSelectBinSum;
		else if (var == "testSelectBinMins") {
			while(!lss.eof()) {
				float binMin = -99999.0f;
				lss >> binMin;
				testSelectBinMins.push_back(binMin);
			}
		}
		else if (var == "testSelectBinMaxes") {
			while(!lss.eof()) {
				float binMax = 99999.0f;
				lss >> binMax;
				testSelectBinMaxes.push_back(binMax);
			}
		}
	}

	if (binnedOutput)
		numBins = (size_t)((binMax - binMin) / binWidth + 1);
	else
		numBins = 1;
}


bool discardInput(float* inputs) {
	if (NUM_INPUTS < 10)
		return false;

	float begAvg = 0;
	for (size_t i = 0; i < 5; i++) {
		begAvg += inputs[i];
	}
	begAvg /= 5;

	float endAvg = 0;
	for (size_t i = 0; i < 5; i++) {
		endAvg += inputs[NUM_INPUTS - i - 1];
	}
	endAvg /= 5;

	return fabs(begAvg - endAvg) > 1;
}

//if overrideBinningSwitch is true, ignores the binnedOutput flag and always uses exact goal
void sampleReadTrainSet(std::string learnsetname, bool discard, size_t* numDiscards, bool overrideBinningSwitch, bool runOnTestSet, size_t startingSample, bool readInBlocks) {
	std::vector<IOPair>* dataset;
	if (runOnTestSet)
		dataset = &testset;
	else
		dataset = &trainset;

	if (numDiscards != NULL) {
		numDiscards[0] = 0;
		numDiscards[1] = 0;
	}
	size_t samplenum = 1;
	(*dataset).clear();
	std::stringstream learnsetss;
	learnsetss << datastring << learnsetname;
	std::ifstream learnset(learnsetss.str().c_str());
	std::string line;
	while (getline(learnset, line)) {
		if (samplenum < startingSample) {
			samplenum++;
			continue;
		}
		else if (readInBlocks && INTERVALS_PER_DATASET > 0 && samplenum >= startingSample + INTERVALS_PER_DATASET)
			break;
		std::stringstream lss(line);
		std::string fname;
		int column;
		int sstart;
		int send;
		float correctoutput;

		lss >> fname;
		lss >> column;
		lss >> sstart;
		lss >> send;
		lss >> correctoutput;

		std::stringstream fullfname;
		fullfname << datastring << fname;
		std::ifstream datafile(fullfname.str().c_str());
		if (!datafile.is_open()) {
			std::stringstream errormsg;
			errormsg << "Couldn't open file " << fullfname.str() << std::endl;
			throw std::runtime_error(errormsg.str().c_str());
		}
		std::string dline;

		std::list<float> inputs;
		for (int i = 1; i <= send && getline(datafile, dline); i++) {
			if (i < sstart)
				continue;

			std::string dum;
			std::stringstream dliness(dline);
			float in;
			for (int j = 0; j<column - 1; j++)
				dliness >> dum;

			dliness >> in;
			inputs.push_back(in);
			if (inputs.size() > NUM_INPUTS)
				inputs.pop_front();

			if (inputs.size() == NUM_INPUTS) {
				IOPair io;
				io.inputs.resize(NUM_INPUTS);
				io.correctoutput = correctoutput / OUTPUT_DIVISOR;
				if (binnedOutput && !overrideBinningSwitch) {
					io.correctbins = getBinnedOutput(correctoutput);
				}
				io.samplenum = samplenum;

				size_t n = 0;
				for (std::list<float>::iterator it = inputs.begin(); it != inputs.end(); it++) {
					io.inputs[n] = *it;
					n++;
				}

				float maxinput = -999999;
				float mininput = 999999;
				for (size_t j = 0; j<NUM_INPUTS; j++) {
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

				if (!discard || !discardInput(&io.inputs[0]))
					(*dataset).push_back(io);
				else if (numDiscards != NULL)
					numDiscards[0]++;

				if (numDiscards != NULL)
					numDiscards[1]++;
			}
		}
		samplenum++;
	}
	learnset.close();
}

float calculateSingleOutput(LayerCollection layers, std::vector<float> inputs) {
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

	checkCudaErrors(cudaMemcpyAsync(d_inputs, &inputs[0], NUM_INPUTS*sizeof(float), cudaMemcpyHostToDevice));

	calculate(layers);

	float output[1];
	checkCudaErrors(cudaMemcpy(output, layers.fixedMat[layers.numFixedNets - 1].outlayer, sizeof(float), cudaMemcpyDeviceToHost));

	return OUTPUT_DIVISOR*output[0];
}

PairedConvCollection createAndInitializePairedConvCollection(size_t numInputs) {
	PairedConvCollection layers;
	layers.conv1 = createLayerCollection(numInputs, CONV_ONLY);
	layers.conv2 = createLayerCollection(numInputs, CONV_ONLY);
	layers.fixed = createLayerCollection(4 * NUM_NEURONS, FIXED_ONLY);

	initializeLayers(&layers.conv1);
	initializeLayers(&layers.conv2);
	initializeLayers(&layers.fixed);

	//link convolution layers to the fixed layer
	FixedNetMatrices* mat = &layers.fixed.fixedMat[0];
	size_t offset = 2 * NUM_NEURONS;
	float* inlayer1;
	float* inlayer2;
	float* inError1;
	float* inError2;

	inlayer1 = mat->inlayer;
	inlayer2 = mat->inlayer + offset;
	inError1 = mat->inErrors;
	inError2 = mat->inErrors + offset;

	if (layers.conv1.mpMat.size() < layers.conv1.numConvolutions) {
		size_t lastConv = layers.conv1.numConvolutions - 1;
		if (layers.conv1.convMat[lastConv].numOutputElements != offset)
			throwLayerLinkError();
		ConvolutionMatrices* convMat = &layers.conv1.convMat[lastConv];
		checkCudaErrors(cudaFree(convMat->outlayer));
		convMat->outlayer = inlayer1;
		checkCudaErrors(cudaFree(convMat->outErrors));
		convMat->outErrors = inError1;
	}
	else {
		size_t lastConv = layers.conv1.numConvolutions - 1;
		if (layers.conv1.mpMat[lastConv].numOutputElements != offset)
			throwLayerLinkError();
		MaxPoolMatrices* mpMat = &layers.conv1.mpMat[lastConv];
		checkCudaErrors(cudaFree(mpMat->outlayer));
		mpMat->outlayer = inlayer1;
		checkCudaErrors(cudaFree(mpMat->outError));
		mpMat->outError = inError1;
	}

	if (layers.conv2.mpMat.size() < layers.conv2.numConvolutions) {
		size_t lastConv = layers.conv2.numConvolutions - 1;
		if (layers.conv2.convMat[lastConv].numOutputElements != offset)
			throwLayerLinkError();
		ConvolutionMatrices* convMat = &layers.conv2.convMat[lastConv];
		checkCudaErrors(cudaFree(convMat->outlayer));
		convMat->outlayer = inlayer2;
		checkCudaErrors(cudaFree(convMat->outErrors));
		convMat->outErrors = inError2;
	}
	else {
		size_t lastConv = layers.conv2.numConvolutions - 1;
		if (layers.conv2.mpMat[lastConv].numOutputElements != offset)
			throwLayerLinkError();
		MaxPoolMatrices* mpMat = &layers.conv2.mpMat[lastConv];
		checkCudaErrors(cudaFree(mpMat->outlayer));
		mpMat->outlayer = inlayer2;
		checkCudaErrors(cudaFree(mpMat->outError));
		mpMat->outError = inError2;
	}

	//unify weights and thresholds in the two convolution layer collections
	for (size_t i = 0; i < layers.conv1.numConvolutions; i++) {
		checkCudaErrors(cudaFree(layers.conv1.convMat[i].weights));
		layers.conv1.convMat[i].weights = layers.conv2.convMat[i].weights;
		checkCudaErrors(cudaFree(layers.conv1.convMat[i].outThresholds));
		layers.conv1.convMat[i].outThresholds = layers.conv2.convMat[i].outThresholds;
	}
	for (size_t i = 0; i < layers.conv1.numFixedNets; i++) {
		checkCudaErrors(cudaFree(layers.conv1.fixedMat[i].weights));
		layers.conv1.fixedMat[i].weights = layers.conv2.fixedMat[i].weights;
		checkCudaErrors(cudaFree(layers.conv1.fixedMat[i].outThresholds));
		layers.conv1.fixedMat[i].outThresholds = layers.conv2.fixedMat[i].outThresholds;
	}

	freeDeviceLayers(&layers.conv1);
	freeDeviceLayers(&layers.conv2);
	freeDeviceLayers(&layers.fixed);

	copyLayersToDevice(&layers.conv1);
	copyLayersToDevice(&layers.conv2);
	copyLayersToDevice(&layers.fixed);

	return layers;
}

bool loadPairedWeights(PairedConvCollection layers, std::string savename) {
	std::stringstream convname;
	convname << savename << "conv";
	//loads twice, but maybe in the future we'll want to disconnect the weights of the two convolutions
	if (!loadWeights(layers.conv1, convname.str()))
		return false;
	if (!loadWeights(layers.conv2, convname.str()))
		return false;
	
	std::stringstream fixedname;
	fixedname << savename << "fixed";
	if (!loadWeights(layers.fixed, fixedname.str()))
		return false;
	return true;
}

void savePairedWeights(PairedConvCollection layers, std::string savename) {
	std::stringstream convname;
	convname << savename << "conv";
	//saves twice, but maybe in the future we'll want to disconnect the weights of the two convolutions
	saveWeights(layers.conv1, convname.str());
	saveWeights(layers.conv2, convname.str());
	
	std::stringstream fixedname;
	fixedname << savename << "fixed";
	saveWeights(layers.fixed, fixedname.str());
}

int getLCType() {
	if (convolutionsOn && fixedNetOn)
		return FULL_NETWORK;
	else if (!convolutionsOn && fixedNetOn)
		return FIXED_ONLY;
	else if (convolutionsOn && !fixedNetOn)
		return CONV_ONLY;
	else {
		std::cout << "Need to have either convolutions or a fixed network (or both)!";
		throw new std::runtime_error("Need to have either convolutions or a fixed network (or both)!");
	}
}

std::vector<float> getBinnedOutput(float output) {
	std::vector<float> bins;

	bins.resize(numBins);
	
	for (size_t i = 0; i < bins.size(); i++) {
		if (output >= binMin + i*binWidth && output < binMin + (i + 1)*binWidth)
			bins[i] = binPositiveOutput;
		else
			bins[i] = binNegativeOutput;
	}
	if (output < binMin)
		bins[0] = binPositiveOutput;
	else if (output > binMin + numBins*binWidth)
		bins[numBins - 1] = binPositiveOutput;

	return bins;
}

void initializeDropoutRNG(LayerCollection* lc) {
	size_t seed = rand();
	size_t sequenceStart = 0;
	for (size_t i = 0; i < lc->numConvolutions; i++) {
		ConvolutionMatrices* lm = &lc->convMat[i];
		ConvolutionParameters* lp = &lc->convPars[i];
		ConvolutionMatrices* d_lm = lc->d_convMat[i];
		ConvolutionParameters* d_lp = lc->d_convPars[i];
		initConvDropoutFactors << <lp->numOutputLocs, lp->numOutputNeurons >> >(d_lm, d_lp, seed, sequenceStart);
		checkCudaErrors(cudaPeekAtLastError());
		sequenceStart += lp->numOutputNeurons;
	}

	seed = rand();
	sequenceStart = 0;
	for (size_t i = 0; i < lc->numFixedNets; i++) {
		FixedNetMatrices* lm = &lc->fixedMat[i];
		FixedNetParameters* lp = &lc->fixedPars[i];
		FixedNetMatrices* d_lm = lc->d_fixedMat[i];
		FixedNetParameters* d_lp = lc->d_fixedPars[i];
		initFixedDropoutFactors << <1, lp->numOutputNeurons >> >(d_lm, d_lp, seed, sequenceStart);
		checkCudaErrors(cudaPeekAtLastError());
		sequenceStart += lp->numOutputNeurons;
	}
}

void generateDropoutMask(LayerCollection* lc) {
	for (size_t i = 0; i < lc->numConvolutions; i++) {
		ConvolutionMatrices* lm = &lc->convMat[i];
		ConvolutionParameters* lp = &lc->convPars[i];
		ConvolutionMatrices* d_lm = lc->d_convMat[i];
		ConvolutionParameters* d_lp = lc->d_convPars[i];
		generateConvDropoutMask << <lp->numOutputLocs, lp->numOutputNeurons >> >(d_lm, d_lp, convDropoutWeight);
	}
	for (size_t i = 0; i < lc->numFixedNets; i++) {
		FixedNetMatrices* lm = &lc->fixedMat[i];
		FixedNetParameters* lp = &lc->fixedPars[i];
		FixedNetMatrices* d_lm = lc->d_fixedMat[i];
		FixedNetParameters* d_lp = lc->d_fixedPars[i];
		generateFixedDropoutMask << <1, lp->numOutputNeurons >> >(d_lm, d_lp, fixedDropoutWeight);
	}
}

void disableDropout() {
	convDropoutWeight = 1.0f;
	fixedDropoutWeight = 1.0f;
}

void enableDropout() {
	convDropoutWeight = savedConvDropoutWeight;
	fixedDropoutWeight = savedFixedDropoutWeight;
}

std::vector<std::vector<IOPair>> getBinnedTrainset() {
	std::vector<std::vector<IOPair>> binset;
	binset.resize(numBins);
	for (size_t i = 0; i < trainset.size(); i++) {
		if (trainset[i].correctoutput < binMin) {
			binset[0].push_back(trainset[i]);
			continue;
		}
		if (trainset[i].correctoutput > binMin + numBins*binWidth) {
			binset[numBins - 1].push_back(trainset[i]);
			continue;
		}
		for (size_t j = 0; j < numBins; j++) {
			if (trainset[i].correctoutput >= binMin + j*binWidth && trainset[i].correctoutput < binMin + (j + 1)*binWidth)
				binset[j].push_back(trainset[i]);
		}
	}
	return binset;
}

std::vector<IOPair>* getTrainSet() {
	return &trainset;
}

std::vector<IOPair>* getTestSet() {
	return &testset;
}

size_t readTwoPriceTrainSet(std::string learnsetname, size_t begin, size_t numIOs, bool overrideBinningSwitch, bool runOnTestSet) {
	std::vector<IOPair>* dataset;
	if (runOnTestSet)
		dataset = &testset;
	else
		dataset = &trainset;
	(*dataset).clear();
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
			io.correctoutput = longprof / OUTPUT_DIVISOR;
			io.secondaryoutput = shortprof / OUTPUT_DIVISOR;

			if (binnedOutput && !overrideBinningSwitch) {
				io.correctbins = getBinnedOutput(longprof);
				io.secondarybins = getBinnedOutput(shortprof);
			}

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
			(*dataset).push_back(io);
		}
	}
	return ionum;
}
