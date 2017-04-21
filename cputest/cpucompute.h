#pragma once
#include <math.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#define TRANSFER_TYPE_RECTIFIER 1
#define TRANSFER_TYPE_SIGMOID 2
#define TRANSFER_TYPE_LINEAR 3

#define TRANSFER_FUNCTION_LIMIT 50.0f
#define TRANSFER_WIDTH 1.0f

#define NEGATIVE_TRANSFER_FACTOR 0.1f

//rofl this class structure
class Layer {
public:
	size_t numInputElements;
	size_t numOutputElements;
	virtual void calc() = 0;
	float* inlayer;
	float* outlayer;
	void Layer::link(Layer* l2);
	virtual void loadWeights(std::ifstream* infile) = 0;
};

class ConvolutionLayer : Layer {
protected:
	float* weights;
	float* outThresholds;
	
	size_t numInputLocs;
	size_t numInputNeurons;
	size_t convSize;
	size_t numOutputLocs;
	size_t numOutputNeurons;

	size_t transferType;

public:
	void calc();
	ConvolutionLayer(size_t numInputNeurons, size_t numInputLocs, size_t numOutputNeurons, size_t numOutputLocs, size_t convSize, size_t transType);
	~ConvolutionLayer();
	void loadWeights(std::ifstream* infile);
};

class FixedLayer : Layer {
protected:
	float* weights;
	float* outThresholds;
	
	size_t numInputNeurons;
	size_t numOutputNeurons;

	size_t transferType;
public:
	void calc();
	FixedLayer(size_t nInputNeurons, size_t nOutputNeurons, size_t transType);
	~FixedLayer();
	void loadWeights(std::ifstream* infile);
};

class MaxPoolLayer : Layer {
protected:
	size_t numInputNeurons;
	size_t numInputLocs;
public:
	void calc();
	MaxPoolLayer(size_t nInputNeurons, size_t nInputLocs);
	~MaxPoolLayer();
	void loadWeights(std::ifstream* infile);
};