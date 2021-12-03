// includes, system
#include "../includes/fft.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cufft.h>
#include "helper.h"
using namespace std;
using namespace faster_dft;
// includes, project
int main(int argc, char **argv)
{
    const int nx = 64;
    const int ny = 40;
    const int nz = 72;
    int SIGNAL_SIZE = nx * ny * nz;
    printf("[3DCUFFT] is starting, (%d, %d, %d) signal size %d...\n", nx, ny, nz, SIGNAL_SIZE);

    cufftDoubleComplex *h_signal=(cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex) * SIGNAL_SIZE);
    for(unsigned int i = 0; i < SIGNAL_SIZE; ++i)
    {
        h_signal[i].x = cos(2.0 / SIGNAL_SIZE * i);
        h_signal[i].y = sin(2.0 / SIGNAL_SIZE * i);
    }

    int mem_size = sizeof(cufftDoubleComplex) * SIGNAL_SIZE;

    // Allocate device memory for signal
    cufftDoubleComplex *d_signal;
    double *d_signal2;
    cudaMalloc((void **)&d_signal, mem_size);
    cudaMalloc((void **)&d_signal2, sizeof(double) * SIGNAL_SIZE);

    // Copy host memory to device
    cudaMemcpy(d_signal, h_signal, mem_size, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    FFT<cufftDoubleComplex> *fft_op = new FFT<cufftDoubleComplex>(nx, ny, nz);
    fft_op->fft3d_all(d_signal2, d_signal, stream);

    const int ite = 10;
    for(int i = 0; i < ite; ++i)
        fft_op->fft3d_all(d_signal2, d_signal, stream);

    struct timeval ss, ee;

    cudaDeviceSynchronize();
    gettimeofday(&ss, NULL);
    for(int i = 0; i < ite; ++i)
        fft_op->fft3d_all(d_signal2, d_signal, stream);
    cudaDeviceSynchronize();
    gettimeofday(&ee, NULL);
    printf("fft3d_all costs %.3f ms\n", diffTime(ss, ee) / ite);

    //Destroy CUFFT context

    // cleanup memory
    free(h_signal);
    cudaFree(d_signal);
}
