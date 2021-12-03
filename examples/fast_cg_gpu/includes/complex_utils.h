#pragma once
// includes, system
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cublas_v2.h>

namespace faster_dft
{

template <typename T>
void print_complex(const T *data, const int size, const char *ss)
{
    T *h_data = (T *)malloc(sizeof(T) * size);
    cudaMemcpy(h_data, data, sizeof(T) * size, cudaMemcpyDeviceToHost);
    printf("%s values\n", ss);
    for(int i = 0; i < size; ++i)
        printf("%d (%lf, %lf)\n", i, (double)h_data[i].x, (double)h_data[i].y);
    printf("\n");
    free(h_data);
}

template <typename T>
void complex_gemm(const T *A, const T *B, T *C, const int M, const int K, const int N, cublasHandle_t handle);

}
