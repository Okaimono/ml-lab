#include "matrix.h"

#include <cmath>

matrix::matrix(u32 rows, u32 cols)
    : rows(rows), cols(cols), data(new float[rows * cols]) {}

matrix::matrix(const matrix& other)
    : rows(other.rows), cols(other.cols), data(new float[other.rows * other.cols]) {
    for (u32 i = 0; i < rows * cols; i++)
        data[i] = other.data[i];
}

matrix& matrix::operator=(const matrix& other) {
    if (this == &other) return *this;
    delete[] data;
    rows = other.rows;
    cols = other.cols;
    data = new float[rows * cols];
    for (u32 i = 0; i < rows * cols; i++)
        data[i] = other.data[i];
    return *this;
}

matrix::~matrix() {
    delete[] data;
}

void matrix::clear() {
    for (u32 i = 0; i < rows * cols; i++) data[i] = 0.0f;
}

void matrix::xavier_init() {
    float scale = sqrtf(2.0f / (rows + cols));

    for (u32 i = 0; i < rows * cols; i++)
        data[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
}

void matrix::relu() {
    for (u32 i = 0; i < rows * cols; i++)
        data[i] = data[i] > 0.0f ? data[i] : 0.0f;
}

void matrix::sigmoid() {
    for (u32 i = 0; i < rows * cols; i++)
        data[i] = 1.0f / (1.0f + expf(-data[i]));
}

void matrix::softmax() {
    float sum = 0.0f;

    for (u32 i = 0; i < rows * cols; i++) {
        data[i] = expf(data[i]);
        sum += data[i];
    }

    for (u32 i = 0; i < rows * cols; i++)
        data[i] /= sum; 
}

matrix matrix::transpose() const {
    matrix result(cols, rows);
    for (u32 i = 0; i < rows; i++)
        for (u32 j = 0; j < cols; j++)
            result.data[j * rows + i] = data[i * cols + j];
    return result;
}

matrix matrix::operator*(const matrix& other) const {
    matrix result(rows, other.cols);
    result.clear();

    for (u32 i = 0; i < rows; i++)
        for (u32 j = 0; j < other.cols; j++)
            for (u32 k = 0; k < cols; k++)
                result.data[i * other.cols + j] +=
                    data[i * cols + k] * other.data[k * other.cols + j];

    return result;
}

matrix matrix::operator+(const matrix& other) const {
    matrix result(rows, cols);

    for (u32 i = 0; i < rows * cols; i++)
        result.data[i] = data[i] + other.data[i];

    return result;
}