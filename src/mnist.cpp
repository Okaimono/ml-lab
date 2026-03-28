#include "mnist.h"
#include <cstdio>
#include <cstdlib>

u32 mnist::reverse_bytes(u32 n) {
    return ((n >> 24) & 0xff)
         | ((n >> 8)  & 0xff00)
         | ((n << 8)  & 0xff0000)
         | ((n << 24) & 0xff000000);
}

void mnist::load(const char* image_path, const char* label_path) {
    // load images
    FILE* img_file = fopen(image_path, "rb");
    
    u32 magic, count, rows, cols;
    fread(&magic, 4, 1, img_file);
    fread(&count, 4, 1, img_file);
    fread(&rows,  4, 1, img_file);
    fread(&cols,  4, 1, img_file);

    count = reverse_bytes(count);
    rows  = reverse_bytes(rows);
    cols  = reverse_bytes(cols);

    for (u32 i = 0; i < count; i++) {
        matrix img(784, 1);
        for (u32 j = 0; j < 784; j++) {
            u8 pixel;
            fread(&pixel, 1, 1, img_file);
            img.data[j] = pixel / 255.0f;
        }
        images.push_back(std::move(img));
    }
    fclose(img_file);

    // load labels
    FILE* lbl_file = fopen(label_path, "rb");

    u32 magic2, count2;
    fread(&magic2, 4, 1, lbl_file);
    fread(&count2, 4, 1, lbl_file);

    for (u32 i = 0; i < count; i++) {
        u8 lbl;
        fread(&lbl, 1, 1, lbl_file);

        matrix one_hot(10, 1);
        one_hot.clear();
        one_hot.data[lbl] = 1.0f;
        labels.push_back(std::move(one_hot));
    }
    fclose(lbl_file);
}

float* mnist::load_data(const char* image_path, int n) {
    FILE* img_file = fopen(image_path, "rb");

    u32 magic, count, rows, cols;
    fread(&magic, 4, 1, img_file);
    fread(&count, 4, 1, img_file);
    fread(&rows,  4, 1, img_file);
    fread(&cols,  4, 1, img_file);

    count = reverse_bytes(count);
    rows  = reverse_bytes(rows);
    cols  = reverse_bytes(cols);

    float* data = (float*)malloc((n * 784) * sizeof(float));
    int index = 0;

    for (u32 i = 0; i < n; i++) {
        float* img = data + (i * 784);
        for (u32 j = 0; j < 784; j++) {
            u8 pixel;
            fread(&pixel, 1, 1, img_file);
            img[j] = pixel / 255.0f;
        }
    }
    fclose(img_file);
    return data;
}

float* mnist::load_labels(const char* label_path, int n) {
    FILE* lbl_file = fopen(label_path, "rb");

    u32 magic2, count2;
    fread(&magic2, 4, 1, lbl_file);
    fread(&count2, 4, 1, lbl_file);

    float* result = (float*)malloc((n * 10) * sizeof(float));
    int index = 0;

    for (u32 i = 0; i < n; i++) {
        u8 lbl;
        fread(&lbl, 1, 1, lbl_file);

        float* one_hot = result + index;
        for (int i = 0; i < 10; i++)
            one_hot[i] = 0;

        one_hot[lbl] = 1.0f;
        index += 10;
    }

    fclose(lbl_file);
    return result;
}