#include "neural_network.h"
#include <iostream>

extern void cuda_relu(float* A, int N);

neural_network::neural_network()
    : W0(128, 784), b0(128, 1), W1(10, 128), b1(10, 1),
      Z0(128, 1), A0(128, 1), Z1(10, 1), A1(10, 1),
      d_A1(10, 1), d_A0(128, 1),
      dW1(10, 128), db1(10, 1),
      dW0(128, 784), db0(128, 1)
{
    W0.xavier_init();
    W1.xavier_init();
    b0.clear();
    b1.clear();
}

matrix neural_network::forward_pass(const matrix& input) {
    matrix Z0 = W0 * input;
    Z0 = b0 + Z0;
    Z0.relu();

    matrix Z1 = W1 * Z0;
    Z1 = b1 + Z1;
    Z1.softmax();

    return Z1;
}

void neural_network::train(const matrix& input, const matrix& y_true, f32 lr) {
    // forward pass
    u32 malloc_size = 0;

    malloc_size += input.rows * input.cols;
    malloc_size += W0.rows * W0.cols;
    malloc_size += W0.rows;
    malloc_size += W1.rows * W1.cols;
    malloc_size += W1.rows;

    Z0 = W0 * input;
    for (u32 i = 0; i < 128; i++) Z0.data[i] += b0.data[i];
    A0 = Z0;
    A0.relu();

    Z1 = W1 * A0;
    for (u32 i = 0; i < 10; i++) Z1.data[i] += b1.data[i];
    A1 = Z1;
    A1.softmax();

    // output error
    for (u32 i = 0; i < 10; i++)
        d_A1.data[i] = A1.data[i] - y_true.data[i];

    // output gradients
    dW1 = d_A1 * A0.transpose();
    db1 = d_A1;

    // hidden error
    d_A0 = W1.transpose() * d_A1;
    for (u32 i = 0; i < 128; i++)
        d_A0.data[i] *= (Z0.data[i] > 0.0f) ? 1.0f : 0.0f;

    // hidden gradients
    dW0 = d_A0 * input.transpose();
    db0 = d_A0;

    // update weights
    for (u32 i = 0; i < W1.rows * W1.cols; i++) W1.data[i] -= lr * dW1.data[i];
    for (u32 i = 0; i < b1.rows * b1.cols; i++) b1.data[i] -= lr * db1.data[i];
    for (u32 i = 0; i < W0.rows * W0.cols; i++) W0.data[i] -= lr * dW0.data[i];
    for (u32 i = 0; i < b0.rows * b0.cols; i++) b0.data[i] -= lr * db0.data[i];
}