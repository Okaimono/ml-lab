#include "mnist.h"
#include "neural_network.h"
#include <cstdio>
#include <iostream>

extern void cuda_mat_mul(float* A, float* B, float* C, int M, int N, int P);

int main() {
    mnist data;
    data.load("../data/train-images-idx3-ubyte", 
              "../data/train-labels-idx1-ubyte");

    neural_network net;
    
    for (u32 epoch = 0; epoch < 10; epoch++) {
        for (u32 i = 0; i < 1000; i++) {
            net.train(data.images[i], data.labels[i], 0.01f);
        }
        printf("epoch %d done\n", epoch);
    }

    for (u32 n = 0; n < 10; n++) {
        matrix predict = net.forward_pass(data.images[n]);

        u32 pred = 0;
        for (u32 i = 1; i < 10; i++)
            if (predict.data[i] > predict.data[pred]) pred = i;

        u32 truth = 0;
        for (u32 i = 0; i < 10; i++)
            if (data.labels[n].data[i] == 1.0f) truth = i;

        printf("predicted: %d, actual: %d\n", pred, truth);
    }
}