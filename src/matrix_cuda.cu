#include "neural_network.h"
#include <stdio.h>

__global__ void mat_mul_kernel(float* A, float* B, float* C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++)
            sum += A[row * N + n] * B[n * P + col];
        C[row * P + col] = sum;
    }
}

__global__ void mat_relu_kernel(float* A, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N) {
        A[row] = A[row] > 0.0f ? A[row] : 0.0f;
    }
}

void cuda_relu(float* A, int N) {
    float* d_A;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + 255) / 256);

    mat_relu_kernel<<<grid, block>>>(d_A, N);

    cudaMemcpy(A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
}

void cuda_mat_mul(float* A, float* B, float* C, int M, int N, int P) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * P * sizeof(float));
    cudaMalloc(&d_C, M * P * sizeof(float));

    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * P * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((P + 15) / 16, (M + 15) / 16);
    mat_mul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, P);

    cudaMemcpy(C, d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

struct GPUBuffers {
    float* W0;
    float* b0;
    float* W1;
    float* b1;
    float* input, y_true;
    float* Z0, *A0, *Z1, *A1;
    float* dA1, *dA0, *dW1, *db1, *dW0, *db0;
};

void cuda_mat_train(neural_network& nn, const matrix& input, const matrix& y_true, f32 lr) {
    GPUBuffers gpu;

    // W0, b0, W1, b1
    cudaMalloc(gpu.W0, nn.W0.rows * nn.W0.cols * sizeof(float));
    cudaMalloc(gpu.b0, nn.b0.rows * nn.b0.cols * sizeof(float));
    cudaMalloc(gpu.W1, nn.W1.rows * nn.W1.cols * sizeof(float));
    cudaMalloc(gpu.b1, nn.b1.rows * nn.b1.cols * sizeof(float));

    cudaMemcpy(gpu.W0, nn.W0.data, nn.W0.rows * nn.W0.cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.b0, nn.b0.data, nn.b0.rows * nn.b0.cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.W1, nn.W1.data, nn.W1.rows * nn.W1.cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.b1, nn.b1.data, nn.b1.rows * nn.b1.cols * sizeof(float), cudaMemcpyHostToDevice);

    // input / y_true
    cudaMalloc(gpu.input, input.rows * input.cols * sizeof(float));
    cudaMalloc(gpu.y_true, y_true.rows * y_true.cols * sizeof(float));

    cudaMemcpy(gpu.input, input.data, input.rows * input.cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu.y_true, y_true.data, y_true.rows * y_true.cols * sizeof(float), cudaMemcpyHostToDevice);

    // Z0, A0, Z1, A1
    cudaMalloc(gpu.Z0, nn.W0.rows * sizeof(float));

    
    



    // u32 malloc_size = 0;

    // malloc_size += input.rows * input.cols;
    // malloc_size += W0.rows * W0.cols;
    // malloc_size += W0.rows;
    // malloc_size += W1.rows * W1.cols;
    // malloc_size += W1.rows;

    // dim3 block(256, 1);
    // dim3 grid((N + 255) / 256, 1);
}