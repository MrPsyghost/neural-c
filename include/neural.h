#ifndef NEURAL_H
#define NEURAL_H

#include <stdbool.h>
#define BARWIDTH 50


typedef struct {
    int in, out;
    double **w;
    double *b;
    double *weightedInputs;
    double *activations;
    double **costGradientW;
    double *costGradientB;
    double *lastInput;
} Layer;

typedef struct {
    int numLayers;
    Layer *layers;
} NeuralNetwork;

// Utilities
double** genWeights(int rows, int cols);
double* genBiases(int cols);

void printMatrix(double **M, int rows, int cols);
void printVector(double *V, int n);
void printOutput(NeuralNetwork *nn, double **X, int n);

void freeLayer(Layer *l);
void freeNN(const NeuralNetwork *nn);
double** make2D(double *data, int rows, int cols);

// Activation Function(s)
double Sigmoid(Layer *l, bool derivative, int i);

// Neural Network
Layer createLayer(const int numNodesIn, const int numNodesOut);
NeuralNetwork createNN(const int numLayers, const int *layerSizes);

double* forward(Layer *l, const double *input);
double** forwardPass(const NeuralNetwork *nn, double **X, int n);

double NodeCost(const Layer *l, double *expectedActivation, int i);
double NodeCostDerivative(const Layer *l, double *expectedActivation, int i);

double* CalculateOutputLayerNodeValues(Layer *l, double *Y, int n);
double* CalculateHiddenLayerNodeValues(Layer *l, Layer* oldLayer, double* oldNodeValues);

double Cost(NeuralNetwork *nn, double **X, double **Y, int n);
double Loss(NeuralNetwork *nn, double **X, double **Y, int n);

void UpdateGradients(Layer *l, double *nodeValues);
void UpdateAllGradients(NeuralNetwork *nn, double **X, double **Y, int n);
void ApplyAllGradients(NeuralNetwork *nn, double lr);

void BackPropagate(NeuralNetwork *nn, double **X, double **Y, int n, double lr, int epochs);

#endif