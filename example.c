#include <stdlib.h>
#include <neural.h>


int main() {
    srand(time(NULL));

    int n = 4;
    const int layerSizes[3] = {2, 2, 1};
    NeuralNetwork nn = createNN(3, layerSizes);
    // XOR Dataset
    double Xraw[4][2] = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
    };
    double Yraw[4][1] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0},
    };
    
    double **X = Tensor(&Xraw[0][0], n, nn.layers[0].out);
    double **Y = Tensor(&Yraw[0][0], n, nn.layers[nn.numLayers - 1].out);
    
    printf("XOR Test:\n");
    printf("X = {\n");
    printMatrix(X, 4, 2);
    printf("}\n\n");
    printf("Y = {\n");
    printMatrix(Y, 4, 1);
    printf("}\n\n");

    printf("Before Training: \n\n");
    printOutput(&nn, X, n);
    
    BackPropagate(&nn, X, Y, n, 0.005, 10000000);
    
    printf("\nAfter Training: \n\n");
    printOutput(&nn, X, n);

    freeNN(&nn);
    free(X);
    free(Y);

    return 0;
}