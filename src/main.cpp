#include "mnist.h"
#include "neural_network.h"
#include <cstdio>
#include <cmath>

#define PIXELS 784

// Gradient geometry transform: any w x h image -> 28x28 gradient -> 784 floats
// Preserves ratio structure, scale invariant, works on any input size
static void gradient(const float* src, int w, int h, float* out) {
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            int l  = c > 0   ? c-1 : c;
            int r2 = c < w-1 ? c+1 : c;
            int u  = r > 0   ? r-1 : r;
            int d  = r < h-1 ? r+1 : r;
            float gx = src[r*w+r2] - src[r*w+l];
            float gy = src[d*w+c]  - src[u*w+c];
            out[r*w+c] = sqrtf(gx*gx + gy*gy);
        }
    }
}

static void resize_bilinear(const float* src, int sw, int sh,
                             float* dst, int dw, int dh) {
    float sx = (float)sw / (float)dw;
    float sy = (float)sh / (float)dh;
    for (int dy = 0; dy < dh; dy++) {
        for (int dx = 0; dx < dw; dx++) {
            float fx = (dx + 0.5f) * sx - 0.5f;
            float fy = (dy + 0.5f) * sy - 0.5f;
            int x0 = (int)fx; int x1 = x0+1;
            int y0 = (int)fy; int y1 = y0+1;
            x0=x0<0?0:x0>=sw?sw-1:x0; x1=x1<0?0:x1>=sw?sw-1:x1;
            y0=y0<0?0:y0>=sh?sh-1:y0; y1=y1<0?0:y1>=sh?sh-1:y1;
            float tx=fx-(int)fx, ty=fy-(int)fy;
            dst[dy*dw+dx] = (1-tx)*(1-ty)*src[y0*sw+x0]
                          +    tx *(1-ty)*src[y0*sw+x1]
                          + (1-tx)*   ty *src[y1*sw+x0]
                          +    tx *   ty *src[y1*sw+x1];
        }
    }
}

// Universal transform: any w x h image -> 784 floats
// 1. compute gradient (ratio structure)
// 2. resize to 28x28
// 3. L2 normalize
static void to_geometry(const float* src, int w, int h, float* out) {
    float* grad = new float[w * h];
    gradient(src, w, h, grad);
    resize_bilinear(grad, w, h, out, 28, 28);
    delete[] grad;

    float norm = 0.0f;
    for (int i = 0; i < PIXELS; i++) norm += out[i]*out[i];
    norm = sqrtf(norm) + 1e-8f;
    for (int i = 0; i < PIXELS; i++) out[i] /= norm;
}

// Synthetic 1 at 100x100 — thick stroke, slight lean, serif
static void draw_one_100(float* img) {
    for (int i = 0; i < 10000; i++) img[i] = 0.0f;
    auto px = [&](int r, int c, float v = 1.0f) {
        if (r >= 0 && r < 100 && c >= 0 && c < 100)
            img[r * 100 + c] = v;
    };
    for (int r = 15; r <= 85; r++) {
        int center = 48 + (r - 15) / 25;
        px(r, center - 1, 0.6f);
        px(r, center,     1.0f);
        px(r, center + 1, 0.6f);
    }
    for (int c = 41; c <= 49; c++) px(15, c, 0.8f);
    for (int c = 44; c <= 55; c++) px(85, c, 0.8f);
}

int main() {
    // 1. load MNIST
    mnist data;
    data.load("../data/train-images-idx3-ubyte",
              "../data/train-labels-idx1-ubyte");

    u32 n_samples = 5000;

    // 2. train on gradient geometry of MNIST (28x28 -> to_geometry -> 784)
    neural_network net;
    printf("Training on gradient geometry...\n");
    for (u32 epoch = 0; epoch < 10; epoch++) {
        for (u32 i = 0; i < n_samples; i++) {
            float z[PIXELS];
            to_geometry(data.images[i].data, 28, 28, z);
            matrix img(PIXELS, 1);
            for (int p = 0; p < PIXELS; p++) img.data[p] = z[p];
            net.train(img, data.labels[i], 0.01f);
        }
        printf("  epoch %d done\n", epoch+1);
    }
    printf("\n");

    // 3. verify on real MNIST
    printf("Verification on MNIST:\n");
    for (u32 n = 0; n < 10; n++) {
        float z[PIXELS];
        to_geometry(data.images[n].data, 28, 28, z);
        matrix img(PIXELS, 1);
        for (int p = 0; p < PIXELS; p++) img.data[p] = z[p];
        matrix pred = net.forward_pass(img);
        int best = 0;
        for (int i = 1; i < 10; i++)
            if (pred.data[i] > pred.data[best]) best = i;
        int truth = 0;
        for (int i = 0; i < 10; i++)
            if (data.labels[n].data[i] == 1.0f) truth = i;
        printf("  predicted: %d | actual: %d -> %s\n",
            best, truth, best == truth ? "CORRECT" : "WRONG");
    }
    printf("\n");

    // 4. synthetic 1 at 100x100 -> to_geometry -> predict
    float raw[10000];
    draw_one_100(raw);
    float z1[PIXELS];
    to_geometry(raw, 100, 100, z1);

    matrix input1(PIXELS, 1);
    for (int i = 0; i < PIXELS; i++) input1.data[i] = z1[i];

    matrix pred1 = net.forward_pass(input1);
    printf("Output probabilities for synthetic '1' (100x100 -> gradient geometry -> predict):\n");
    for (int i = 0; i < 10; i++)
        printf("  class %d: %.4f\n", i, pred1.data[i]);

    int best1 = 0;
    for (int i = 1; i < 10; i++)
        if (pred1.data[i] > pred1.data[best1]) best1 = i;
    printf("\nPredicted: %d | Actual: 1 -> %s\n",
        best1, best1 == 1 ? "CORRECT" : "WRONG");

    return 0;
}

// #include "mnist.h"
// #include "neural_network.h"
// #include <cstdio>
// #include <iostream>

// #include <chrono>

// extern void cuda_mat_mul(float* A, float* B, float* C, int M, int N, int P);
// extern void cuda_mat_train(matrix& W0, matrix& b0, matrix& W1, matrix& b1);

// void test(neural_network& net);
// void train_mnist();

// int main() {
    
// }

// void test(neural_network& net) {
//     mnist data;
//     data.load("../data/train-images-idx3-ubyte", 
//               "../data/train-labels-idx1-ubyte");

//     for (u32 n = 0; n < 10; n++) {
//         matrix predict = net.forward_pass(data.images[n]);

//         u32 pred = 0;
//         for (u32 i = 1; i < 10; i++)
//             if (predict.data[i] > predict.data[pred]) pred = i;

//         u32 truth = 0;
//         for (u32 i = 0; i < 10; i++)
//             if (data.labels[n].data[i] == 1.0f) truth = i;

//         printf("predicted: %d, actual: %d\n", pred, truth);
//     }
// }

// void train_mnist() {
//     mnist data;
//     data.load("../data/train-images-idx3-ubyte", 
//               "../data/train-labels-idx1-ubyte");

//     neural_network net;
    
//     for (u32 epoch = 0; epoch < 10; epoch++) {
//         for (u32 i = 0; i < 1000; i++) {
//             net.train(data.images[i], data.labels[i], 0.01f);
//         }
//         printf("epoch %d done\n", epoch);
//     }

//     for (u32 n = 0; n < 10; n++) {
//         matrix predict = net.forward_pass(data.images[n]);

//         u32 pred = 0;
//         for (u32 i = 1; i < 10; i++)
//             if (predict.data[i] > predict.data[pred]) pred = i;

//         u32 truth = 0;
//         for (u32 i = 0; i < 10; i++)
//             if (data.labels[n].data[i] == 1.0f) truth = i;

//         printf("predicted: %d, actual: %d\n", pred, truth);
//     }
// }