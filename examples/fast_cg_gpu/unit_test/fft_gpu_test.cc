#include "../includes/fft.h"
#include "helper.h"
#include "complex.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
using namespace std;
using namespace faster_dft;

bool inverse = false ;
const int N = 2048;
#define PI 3.141592653589793

typedef double T;

int main()
{
    T *cr;
    T *ci;
    T *wn_r;
    T *wn_i;
    cudaMalloc((void **)&cr, sizeof(T) * N);
    cudaMalloc((void **)&ci, sizeof(T) * N);
    cudaMalloc((void **)&wn_r, sizeof(T) * N);
    cudaMalloc((void **)&wn_i, sizeof(T) * N);

    T *ca_cpu = (T *)malloc(sizeof(T) * N);
    T *cb_cpu = (T *)malloc(sizeof(T) * N);
    for(int i = 0; i < N; ++i)
    {
        ca_cpu[i] = cos(2.0 / N * i);
        cb_cpu[i] = sin(2.0 / N * i);
    }
    cudaMemcpy(cr, ca_cpu, sizeof(T) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(ci, cb_cpu, sizeof(T) * N, cudaMemcpyHostToDevice);

    wn_init(wn_r, wn_i, N);

    for(int i = 0; i < 1; ++i)
        fft_kernel_launcher(wn_r, wn_i, cr, ci, N);

    cudaMemcpy(ca_cpu, cr, sizeof(T) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(cb_cpu, ci, sizeof(T) * N, cudaMemcpyDeviceToHost);
    for(int i = 0; i < 10; ++i)
        cout << i << " " << ca_cpu[i] << " " << cb_cpu[i] << endl;

    int ite = 10;
    struct timeval ss, ee;
    cudaDeviceSynchronize();
    gettimeofday(&ss, NULL);
    for(int i = 0; i < ite; ++i)
        fft_kernel_launcher(wn_r, wn_i, cr, ci, N);
    cudaDeviceSynchronize();
    gettimeofday(&ee, NULL);
    printf("fft %d eles costs %.3f ms\n", N, diffTime(ss, ee) / ite);
}
