#include "mnist.h"
#include "neural_network.h"
#include <cstdio>
#include <iostream>
#include <chrono>

extern void cuda_mat_mul(float* A, float* B, float* C, int M, int N, int P);
extern void cuda_mat_train(matrix& W0, matrix& b0, matrix& W1, matrix& b1);

void test(neural_network& net, mnist& data);

int main() {
    mnist data;
    data.load("../data/train-images-idx3-ubyte",
              "../data/train-labels-idx1-ubyte");

    // CPU benchmark
    printf("=== CPU 1 epoch 60k ===\n");
    neural_network cpu_net;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (u32 i = 0; i < 60000; i++)
        cpu_net.train(data.images[i], data.labels[i], 0.01f);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    printf("CPU time: %.2f ms (%.2f sec)\n", cpu_ms, cpu_ms / 1000.0);
    test(cpu_net, data);

    // CUDA benchmark
    printf("\n=== CUDA 1 epoch 60k ===\n");
    neural_network cuda_net;

    auto cuda_start = std::chrono::high_resolution_clock::now();
    cuda_mat_train(cuda_net.W0, cuda_net.b0, cuda_net.W1, cuda_net.b1);
    auto cuda_end = std::chrono::high_resolution_clock::now();

    double cuda_ms = std::chrono::duration<double, std::milli>(cuda_end - cuda_start).count();
    printf("CUDA time: %.2f ms (%.2f sec)\n", cuda_ms, cuda_ms / 1000.0);
    test(cuda_net, data);

    printf("\n=== SPEEDUP ===\n");
    printf("%.2fx faster\n", cpu_ms / cuda_ms);
}

void test(neural_network& net, mnist& data) {
    u32 correct = 0;
    u32 total = 1000;

    for (u32 n = 0; n < total; n++) {
        matrix predict = net.forward_pass(data.images[n]);

        u32 pred = 0;
        for (u32 i = 1; i < 10; i++)
            if (predict.data[i] > predict.data[pred]) pred = i;

        u32 truth = 0;
        for (u32 i = 0; i < 10; i++)
            if (data.labels[n].data[i] == 1.0f) truth = i;

        if (pred == truth) correct++;
    }

    printf("accuracy: %d/%d (%.1f%%)\n", correct, total, (float)correct / total * 100.0f);
}