/*
* Author: Xiaoying Jia
* Project: FasterDFT
* Department: ByteDance Data-AML
* Email: {jiaxiaoying}@bytedance.com
*/
#include "includes/fft.h"
#include "includes/common.h"
#include "includes/complex_utils.h"
#include <cufft.h>
#include <cmath>
using namespace std;

#define WARP_SIZE  32
#define PI 3.141592653589793

namespace faster_dft
{
template <typename ComplexType, typename ElemType>
__global__
void data_vreff_dot_kernel(const ElemType *vr_eff, ComplexType *data, const int N, const int K)
{
    int total_threads = blockDim.x * gridDim.x;

	for(int idx = 0; idx < K; idx++)
	{
		for(int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < N; tid += total_threads)
		{
			ComplexType data_val = data[idx * N + tid];
			ElemType vr_eff_val = __ldg(&vr_eff[tid]);
			data_val.x *= vr_eff_val;
			data_val.y *= vr_eff_val;
			data[idx * N + tid] = data_val;
		}
	}
}

template <typename T>
__global__
void data_scale_kernel(T *data, const int N, const int K, const double scaler)
{
    int total_threads = blockDim.x * gridDim.x;
	for(int idx = 0; idx < K; idx++)
	{
		for(int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < N; tid += total_threads)
		{
			T data_val = data[idx * N + tid];
			data_val.x *= scaler;
			data_val.y *= scaler;
			data[idx * N + tid] = data_val;
		}
	}
}


template <OperationType OpType>
void FFT<OpType>::fft3d_all(const ElemType *vr_eff, ComplexType *data, cudaStream_t stream)
{
    cufftSetStream(plan_, stream);

    if(OpType == OperationType::FP64)
        cufftExecZ2Z(plan_, (cuDoubleComplex *)data, (cuDoubleComplex *)data, CUFFT_INVERSE);
    else
        cufftExecC2C(plan_, (cuComplex *)data, (cuComplex *)data, CUFFT_INVERSE);

    dim3 grid;
    dim3 block(512);
    grid.x = (N_ - 1) / block.x + 1;

    data_vreff_dot_kernel<ComplexType, ElemType><<<grid, block, 0, stream>>>(vr_eff, data, N_, batch_);

    if(OpType == OperationType::FP64)
        cufftExecZ2Z(plan_, (cuDoubleComplex *)data, (cuDoubleComplex *)data, CUFFT_FORWARD);
    else
        cufftExecC2C(plan_, (cuComplex *)data, (cuComplex *)data, CUFFT_FORWARD);

    double scaler = 1.0 / (nx_ * ny_ * nz_);
    data_scale_kernel<<<grid, block, 0, stream>>>(data, N_, batch_, scaler);
}

template void FFT<OperationType::FP64>::fft3d_all(const double *vreff, cuDoubleComplex *data, cudaStream_t stream);
template void FFT<OperationType::FP32>::fft3d_all(const float *vreff, cuComplex *data, cudaStream_t stream);
}

