#pragma once
#include "types.h"
#include "matrix.h"
#include <vector>

struct mnist {
    std::vector<matrix> images;
    std::vector<matrix> labels;

    void load(const char* image_path, const char* label_path);

private:
    u32 reverse_bytes(u32 n);
};