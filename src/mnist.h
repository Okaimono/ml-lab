#pragma once
#include "types.h"
#include "matrix.h"
#include <vector>

struct mnist {
    std::vector<matrix> images;
    std::vector<matrix> labels;

    void load(const char* image_path, const char* label_path);

    static float* load_data(const char* image_path, int n);
    static float* load_labels(const char* label_path, int n);

private:
    static u32 reverse_bytes(u32 n);
};