#pragma once
#include "matrix.h"

struct neural_network {
    neural_network();

    matrix W0;
    matrix b0;
    matrix W1;
    matrix b1;

    matrix Z0, A0, Z1, A1;
    matrix d_A1, d_A0, dW1, db1, dW0, db0;

    matrix forward_pass(const matrix& input);
    void train(const matrix& input, const matrix& y_true, f32 lr);
};