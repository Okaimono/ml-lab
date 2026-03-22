#include <stdio.h>

void cuda_check() {
    int count;
    cudaGetDeviceCount(&count);
    printf("CUDA devices: %d\n", count);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("VRAM: %zu MB\n", prop.totalGlobalMem / 1024 / 1024);
}