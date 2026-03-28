#include "mnist.h"
#include "neural_network.h"
#include <stdio.h>
#include <string.h>
#include <vector>

__global__ void mat_mul_kernel(float* A, float* B, float* C, int M, int N, int P) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < P) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++)
            sum += A[row * N + n] * B[n * P + col];
        C[row * P + col] = sum;
    }
}

__global__ void mat_add_kernel(float* A, float* B, float* C, bool add, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = add ? A[idx] + B[idx] : A[idx] - B[idx];
}

__global__ void mat_relu_kernel(float* A, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    C[row] = A[row] > 0.0f ? A[row] : 0.0f;
}

__global__ void mat_softmax_kernel(float* A, float* out, int n) {
    int i = threadIdx.x;
    if (i < n) {
        __shared__ float d_sum;

        if (i == 0) d_sum = 0;
        __syncthreads();

        atomicAdd(&d_sum, expf(A[i]));
        __syncthreads();

        out[i] = expf(A[i]) / d_sum;
    }
}

__global__ void mat_drelu_kernel(float* Z, float* dA, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = dA[i] * (Z[i] > 0.0f ? 1.0f : 0.0f);
}

__global__ void mat_transpose_kernel(float* A, float* out, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        out[col * M + row] = A[row * N + col];
    }
}

__global__ void mat_update_kernel(float* W, float* grad, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) W[i] -= lr * grad[i];
}

void cuda_mat_train(matrix& W0, matrix& b0, matrix& W1, matrix& b1) {
    const int count = 60000;
    float* data   = mnist::load_data("../data/train-images-idx3-ubyte", count);
    float* labels = mnist::load_labels("../data/train-labels-idx1-ubyte", count);

    float* d_labels;
    float* d_data;

    float* d_W0;
    float* d_b0;
    float* d_W1;
    float* d_b1;

    float* d_Z0;
    float* d_A0;
    float* d_Z1;
    float* d_A1;

    float* d_A1_err;
    float* d_W1_grad;
    float* d_b1_grad;
    float* d_A0_err;
    float* d_W0_grad;
    float* d_b0_grad;

    float* d_A0_T;
    float* d_W1_T;
    float* d_input_T;

    // Allocate
    cudaMalloc(&d_labels,    10  * count * sizeof(float));
    cudaMalloc(&d_data,      784 * count * sizeof(float));

    cudaMalloc(&d_W0,        128 * 784 * sizeof(float));
    cudaMalloc(&d_b0,        128 * 1   * sizeof(float));
    cudaMalloc(&d_W1,        10  * 128 * sizeof(float));
    cudaMalloc(&d_b1,        10  * 1   * sizeof(float));

    cudaMalloc(&d_Z0,        128 * 1   * sizeof(float));
    cudaMalloc(&d_A0,        128 * 1   * sizeof(float));
    cudaMalloc(&d_Z1,        10  * 1   * sizeof(float));
    cudaMalloc(&d_A1,        10  * 1   * sizeof(float));

    cudaMalloc(&d_A1_err,    10  * 1   * sizeof(float));
    cudaMalloc(&d_W1_grad,   10  * 128 * sizeof(float));
    cudaMalloc(&d_b1_grad,   10  * 1   * sizeof(float));
    cudaMalloc(&d_A0_err,    128 * 1   * sizeof(float));
    cudaMalloc(&d_W0_grad,   128 * 784 * sizeof(float));
    cudaMalloc(&d_b0_grad,   128 * 1   * sizeof(float));

    cudaMalloc(&d_A0_T,      128 * 1   * sizeof(float));
    cudaMalloc(&d_W1_T,      128 * 10  * sizeof(float));
    cudaMalloc(&d_input_T,   784 * 1   * sizeof(float));

    // Copy data and weights to GPU
    cudaMemcpy(d_labels, labels,   10  * count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data,   data,     784 * count * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_W0, W0.data, 128 * 784 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b0, b0.data, 128 * 1   * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1.data, 10  * 128 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1.data, 10  * 1   * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid(1);
    dim3 mul_block(16, 16);
    dim3 mul_grid(1, 1);

    int M, N;
    float lr = 0.01f;

    for (int i = 0; i < count; i++) {
        float* d_input = d_data   + (i * 784);
        float* d_ytrue = d_labels + (i * 10);

        // FORWARD PASS

        // Z0 = W0 * input  (128x784) * (784x1) = (128x1)
        M = 128; N = 1;
        mul_grid = dim3((M + 15) / 16, (N + 15) / 16);
        mat_mul_kernel<<<mul_grid, mul_block>>>(d_W0, d_input, d_Z0, 128, 784, 1);
        mat_add_kernel<<<grid, block>>>(d_b0, d_Z0, d_Z0, true, 128);
        mat_relu_kernel<<<grid, block>>>(d_Z0, d_A0);

        // Z1 = W1 * A0  (10x128) * (128x1) = (10x1)
        M = 10; N = 1;
        mul_grid = dim3((M + 15) / 16, (N + 15) / 16);
        mat_mul_kernel<<<mul_grid, mul_block>>>(d_W1, d_A0, d_Z1, 10, 128, 1);
        mat_add_kernel<<<grid, block>>>(d_b1, d_Z1, d_Z1, true, 10);
        mat_softmax_kernel<<<grid, block>>>(d_Z1, d_A1, 10);

        // BACKPROP

        // dA1_err = A1 - ytrue  (10x1)
        mat_add_kernel<<<grid, block>>>(d_A1, d_ytrue, d_A1_err, false, 10);

        // A0_T = A0^T  (128x1) -> (1x128)
        M = 128; N = 1;
        mul_grid = dim3((M + 15) / 16, (N + 15) / 16);
        mat_transpose_kernel<<<mul_grid, mul_block>>>(d_A0, d_A0_T, 128, 1);

        // dW1_grad = dA1_err * A0_T  (10x1) * (1x128) = (10x128)
        M = 10; N = 128;
        mul_grid = dim3((M + 15) / 16, (N + 15) / 16);
        mat_mul_kernel<<<mul_grid, mul_block>>>(d_A1_err, d_A0_T, d_W1_grad, 10, 1, 128);

        // db1_grad = dA1_err  (10x1)
        cudaMemcpy(d_b1_grad, d_A1_err, 10 * sizeof(float), cudaMemcpyDeviceToDevice);

        // W1_T = W1^T  (10x128) -> (128x10)
        M = 10; N = 128;
        mul_grid = dim3((M + 15) / 16, (N + 15) / 16);
        mat_transpose_kernel<<<mul_grid, mul_block>>>(d_W1, d_W1_T, 10, 128);

        // dA0_err = W1_T * dA1_err  (128x10) * (10x1) = (128x1)
        M = 128; N = 1;
        mul_grid = dim3((M + 15) / 16, (N + 15) / 16);
        mat_mul_kernel<<<mul_grid, mul_block>>>(d_W1_T, d_A1_err, d_A0_err, 128, 10, 1);

        // drelu mask on dA0_err using Z0
        mat_drelu_kernel<<<grid, block>>>(d_Z0, d_A0_err, d_A0_err, 128);

        // input_T = input^T  (784x1) -> (1x784)
        M = 784; N = 1;
        mul_grid = dim3((M + 15) / 16, (N + 15) / 16);
        mat_transpose_kernel<<<mul_grid, mul_block>>>(d_input, d_input_T, 784, 1);

        // dW0_grad = dA0_err * input_T  (128x1) * (1x784) = (128x784)
        M = 128; N = 784;
        mul_grid = dim3((M + 15) / 16, (N + 15) / 16);
        mat_mul_kernel<<<mul_grid, mul_block>>>(d_A0_err, d_input_T, d_W0_grad, 128, 1, 784);

        // db0_grad = dA0_err  (128x1)
        cudaMemcpy(d_b0_grad, d_A0_err, 128 * sizeof(float), cudaMemcpyDeviceToDevice);

        // UPDATE WEIGHTS
        mat_update_kernel<<<dim3((10*128+127)/128), block>>>(d_W1, d_W1_grad, lr, 10*128);
        mat_update_kernel<<<dim3((10+127)/128),     block>>>(d_b1, d_b1_grad, lr, 10);
        mat_update_kernel<<<dim3((128*784+127)/128),block>>>(d_W0, d_W0_grad, lr, 128*784);
        mat_update_kernel<<<dim3((128+127)/128),    block>>>(d_b0, d_b0_grad, lr, 128);
    }

    cudaMemcpy(W0.data, d_W0, 128 * 784 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b0.data, d_b0, 128 * 1   * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(W1.data, d_W1, 10  * 128 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b1.data, d_b1, 10  * 1   * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Training complete.\n");

    cudaFree(d_labels);
    cudaFree(d_data);
    cudaFree(d_W0);
    cudaFree(d_b0);
    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_Z0);
    cudaFree(d_A0);
    cudaFree(d_Z1);
    cudaFree(d_A1);
    cudaFree(d_A1_err);
    cudaFree(d_W1_grad);
    cudaFree(d_b1_grad);
    cudaFree(d_A0_err);
    cudaFree(d_W0_grad);
    cudaFree(d_b0_grad);
    cudaFree(d_A0_T);
    cudaFree(d_W1_T);
    cudaFree(d_input_T);
}

void cuda_relu(float* A, int N) {
    // float* d_A;
    // cudaMalloc(&d_A, N * sizeof(float));
    // cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);

    // dim3 block(256);
    // dim3 grid((N + 255) / 256);

    //mat_relu_kernel<<<grid, block>>>(d_A, N);

    // cudaMemcpy(A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost);

    // cudaFree(d_A);
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




    // neural_network nn;

    // u32 size = 0;

    // size += nn.W0.rows * nn.W0.cols;      // W0
    // size += nn.b0.rows;                   // b0
    // size += nn.W1.rows * nn.W1.cols;      // W1
    // size += nn.b1.rows;                   // b1

    // size += nn.W0.rows;                   // Z0 (output of W0 * input)
    // size += nn.W0.rows;                   // A0 (relu of Z0)
    // size += nn.W1.rows;                   // Z1 (output of W1 * A0)
    // size += nn.W1.rows;                   // A1 (relu of Z1)

    // size += nn.W1.rows;                   // dA1
    // size += nn.W0.rows;                   // dA0
    // size += nn.W1.rows * nn.W1.cols;      // dW1
    // size += nn.b1.rows;                   // db1
    // size += nn.W0.rows * nn.W0.cols;      // dW0
    // size += nn.b0.rows;                   // db0

    // float* d_arena;
    // cudaMalloc(&d_arena, size * sizeof(float));

    // // carve up the arena
    // GPUBuffers gpu;
    // float* ptr = d_arena;

    // gpu.W0 = ptr; ptr += nn.W0.rows * nn.W0.cols;
    // gpu.b0 = ptr; ptr += nn.b0.rows;
    // gpu.W1 = ptr; ptr += nn.W1.rows * nn.W1.cols;
    // gpu.b1 = ptr; ptr += nn.b1.rows;

    // gpu.Z0 = ptr; ptr += nn.W0.rows;
    // gpu.A0 = ptr; ptr += nn.W0.rows;
    // gpu.Z1 = ptr; ptr += nn.W1.rows;
    // gpu.A1 = ptr; ptr += nn.W1.rows;

    // gpu.dA1 = ptr; ptr += nn.W1.rows;
    // gpu.dA0 = ptr; ptr += nn.W0.rows;
    // gpu.dW1 = ptr; ptr += nn.W1.rows * nn.W1.cols;
    // gpu.db1 = ptr; ptr += nn.b1.rows;
    // gpu.dW0 = ptr; ptr += nn.W0.rows * nn.W0.cols;
    // gpu.db0 = ptr; ptr += nn.b0.rows;


