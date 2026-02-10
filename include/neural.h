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
void printMatrix(double **M, int rows, int cols);
void printVector(double *V, int n);
void printOutput(NeuralNetwork *nn, double **X, int n);

void freeLayer(Layer *l);
void freeNN(const NeuralNetwork *nn);
double** Tensor(double *data, int rows, int cols);

// Activation Function(s)
double Sigmoid(Layer *l, bool derivative, int i);

// Neural Network
NeuralNetwork createNN(const int numLayers, const int *layerSizes);

double** forwardPass(const NeuralNetwork *nn, double **X, int n);

void BackPropagate(NeuralNetwork *nn, double **X, double **Y, int n, double lr, int epochs);

#endif