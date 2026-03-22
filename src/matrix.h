#pragma once
#include "types.h"

struct matrix {
    matrix(u32 rows, u32 cols);
    matrix(const matrix& other);
    matrix& operator=(const matrix& other);
    ~matrix();

    u32 rows;
    u32 cols;
    f32* data;

    void xavier_init();
    void clear();

    void relu();
    void sigmoid();
    void softmax();

    matrix transpose() const;

    matrix operator*(const matrix& other) const;
    matrix operator+(const matrix& other) const;
};