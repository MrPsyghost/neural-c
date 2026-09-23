#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "neural.h"

static inline double rand_uniform(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

static inline double** genWeights(int rows, int cols) {
    // rows x cols
    
    double **w = malloc(rows * sizeof(double*));
    if (!w) return NULL;
    
    for (int y = 0; y < rows; y++) {
        w[y] = (double*) malloc(cols * sizeof(double));
        
        if (!w[y]) { for (int x = 0; x < y; x++) free(w[x]); free(w); return NULL; }
        
        for (int x = 0; x < cols; x++) {
            w[y][x] = rand_uniform(-2.0, 2.0);
        }
    }
    
    return w;
}

static inline double* genBiases(int cols) {
    // cols
    
    double *b = malloc(cols * sizeof(double));
    if (!b) return NULL;
    
    for (int i = 0; i < cols; i++) {
        b[i] = rand_uniform(-2.0, 2.0);
    }
    
    return b;
}


void printMatrix(double **M, int rows, int cols) {
    for (int y = 0; y < rows; y++) {
        printf("{");
        for (int x = 0; x < cols; x++) {
            printf("%f ", M[y][x]);
        }
        printf("}\n");
    }
}

void printVector(double *V, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f \n", V[i]);
    }
}

void printOutput(NeuralNetwork *nn, double **X, int n) {
    double **output = forwardPass(nn, X, n);
    for (int i = 0; i < n; i++) {
        printVector(output[i], nn->layers[nn->numLayers - 1].out);
        free(output[i]);
    }
    free(output);
}


void freeLayer(Layer *l) {
    for (int i = 0; i < l->in; i++) {
        free(l->w[i]);
        free(l->costGradientW[i]);
    }

    free(l->w);
    free(l->b);
    free(l->weightedInputs);
    free(l->activations);
    free(l->costGradientB);
    free(l->costGradientW);
    free(l->lastInput);
}

void freeNN(const NeuralNetwork *nn) {
    for (int i = 0; i < nn->numLayers; i++) {
        freeLayer(&nn->layers[i]);
    }
    free(nn->layers);
}


double** Tensor(double *data, int rows, int cols) {
    double **arr = malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++)
        arr[i] = data + i * cols;
    return arr;
}




double Sigmoid(Layer *l, bool derivative, int i) {
    double activation = 1 / (1 + exp(-l->weightedInputs[i]));
    if (!derivative) {
        return activation;
    } else {
        return activation * (1 - activation);
    }
}




static inline Layer createLayer(const int numNodesIn, const int numNodesOut) {
    Layer l;
    l.in = numNodesIn;
    l.out = numNodesOut;
    l.w = genWeights(numNodesIn, numNodesOut);
    l.b = genBiases(numNodesOut);
    l.weightedInputs = malloc(numNodesOut * sizeof(double));
    l.activations = malloc(numNodesOut * sizeof(double));
    l.costGradientB = calloc(numNodesOut, sizeof(double));
    l.costGradientW = malloc(numNodesIn * sizeof(double*));
    for (int i = 0; i < numNodesIn; i++) {
        l.costGradientW[i] = calloc(numNodesOut, sizeof(double));
    }
    l.lastInput = malloc(numNodesIn * sizeof(double));
    return l;
}

NeuralNetwork createNN(const int numLayers, const int *layerSizes) {
    NeuralNetwork nn;
    nn.numLayers = numLayers - 1;
    nn.layers = malloc(nn.numLayers * sizeof(Layer));
    for (int i = 1; i < numLayers; i++) {
        nn.layers[i - 1] = createLayer(layerSizes[i - 1], layerSizes[i]);
    }
    return nn;
}


static inline double* forward(Layer *l, const double *input) {
    memcpy(l->lastInput, input, l->in * sizeof(double));

    for (int j = 0; j < l->out; j++) {
        double z = l->b[j];

        for (int i = 0; i < l->in; i++) {
            z += input[i] * l->w[i][j];
        }

        l->weightedInputs[j] = z;
        l->activations[j] = Sigmoid(l, false, j);
    }

    double *copy = malloc(l->out * sizeof(double));
    memcpy(copy, l->activations, l->out * sizeof(double));
    return copy;
}

double** forwardPass(const NeuralNetwork *nn, double **X, int n) {
    double **output = X;
    for (int j = 0; j < nn->numLayers; j++) {
        double **next = malloc(n * sizeof(double*));
        for (int i = 0; i < n; i++) {
            next[i] = forward(&nn->layers[j], output[i]);
        }
        if (j > 0) { free(output); }
        output = next;
    }
    return output;
}


static inline double NodeCost(const Layer *l, double *expectedActivation, int i) {
    double error = pow(l->activations[i] - expectedActivation[i], 2);
    return error;
}

static inline double NodeCostDerivative(const Layer *l, double *expectedActivation, int i) {
    double error = 2 * (l->activations[i] - expectedActivation[i]);
    return error;
}


static inline double* CalculateOutputLayerNodeValues(Layer *l, double *Y, int n) {
    double *nodeValues = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double costDerivative = NodeCostDerivative(l, Y, i);
        double activationDerivative = Sigmoid(l, true, i);
        nodeValues[i] = activationDerivative * costDerivative;
    }
    return nodeValues;
}

static inline double* CalculateHiddenLayerNodeValues(Layer *l, Layer* oldLayer, double* oldNodeValues) {
    double* newNodeValues = calloc(l->out, sizeof(double));
    for (int newNodeIndex = 0; newNodeIndex < l->out; newNodeIndex++) {
        double newNodeValue = 0.0;
        for (int oldNodeIndex = 0; oldNodeIndex < oldLayer->out; oldNodeIndex++) {
            newNodeValue += oldLayer->w[newNodeIndex][oldNodeIndex] * oldNodeValues[oldNodeIndex];
        }
        newNodeValues[newNodeIndex] = newNodeValue * Sigmoid(l, true, newNodeIndex);
    }
    return newNodeValues;
}


static inline double Cost(NeuralNetwork *nn, double **X, double **Y, int n) {
    forwardPass(nn, X, n);
    Layer *outputLayer = &nn->layers[nn->numLayers - 1];
    double cost = 0;
    for (int i = 0; i < n; i++) {
        for (int nodeOut = 0; nodeOut < nn->layers[nn->numLayers - 1].out ; nodeOut++) {
            cost += NodeCost(outputLayer, Y[i], nodeOut);
        }
    }
    return cost;
}

static inline double Loss(NeuralNetwork *nn, double **X, double **Y, int n) {
    double totalCost = 0;
    totalCost += Cost(nn, X, Y, n);
    return (totalCost / n);
}


static inline void UpdateGradients(Layer *l, double *nodeValues) {
    for (int nodeOut = 0; nodeOut < l->out; nodeOut++) {
        l->costGradientB[nodeOut] += nodeValues[nodeOut];
        for (int nodeIn = 0; nodeIn < l->in; nodeIn++) {
            l->costGradientW[nodeIn][nodeOut] += l->lastInput[nodeIn] * nodeValues[nodeOut];
        }
    }
}

static inline void UpdateAllGradients(NeuralNetwork *nn, double **X, double **Y, int n) {
    for (int i = 0; i < n; i++) {
        forward(&nn->layers[0], X[i]);
        for (int l = 1; l < nn->numLayers; l++) {
            forward(&nn->layers[l], nn->layers[l-1].activations);
        }
        Layer *outputLayer = &nn->layers[nn->numLayers - 1];
        double *nodeValues = CalculateOutputLayerNodeValues(outputLayer, Y[i], nn->layers[nn->numLayers - 1].out );
        UpdateGradients(outputLayer, nodeValues);
        for (int hiddenLayerIndex = (nn->numLayers - 1 - 1); hiddenLayerIndex > -1; hiddenLayerIndex--) {
            Layer *hiddenLayer = &nn->layers[hiddenLayerIndex];
            double *prev = nodeValues;
            nodeValues = CalculateHiddenLayerNodeValues(hiddenLayer, &nn->layers[hiddenLayerIndex + 1], nodeValues);
            free(prev);
            UpdateGradients(hiddenLayer, nodeValues);
        }
        free(nodeValues);
    }
}

static inline void ApplyAllGradients(NeuralNetwork *nn, double lr) {
    for (int i = 0; i < nn->numLayers; i++) {
        Layer *l = &nn->layers[i];
        for (int in = 0; in < l->in; in++) {
            for (int out = 0; out < l->out; out++) {
                l->w[in][out] -= lr * l->costGradientW[in][out]; 
                l->costGradientW[in][out] = 0.0; 
            }
        }
        for (int out = 0; out < l->out; out++) {
            l->b[out] -= lr * l->costGradientB[out]; 
            l->costGradientB[out] = 0.0; 
        }
    }
}


void BackPropagate(NeuralNetwork *nn, double **X, double **Y, int n, double lr, int epochs) {
    printf("\nTraining: |");
    int last = 0;
    for (int epoch = 0; epoch < epochs; epoch++) {
        UpdateAllGradients(nn, X, Y, n);
        ApplyAllGradients(nn, lr);
        int progress = ((epoch + 1) * BARWIDTH) / epochs;
        while (progress > last) {
            fflush(stdout);
            printf("#");
            last++;
        }
    }
    printf("|\n");
}