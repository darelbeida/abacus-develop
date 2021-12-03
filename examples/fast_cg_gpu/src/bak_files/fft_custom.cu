/*
* Author: Xiaoying Jia
* Project: FFT
* Department: ByteDance Data-AML
* Email: {jiaxiaoying}@bytedance.com
*/
#include "../includes/fft.h"
#include <cufft.h>
#include <cmath>
using namespace std;

#define WARP_SIZE  32
#define PI 3.141592653589793

namespace faster_dft
{

template <typename T>
void wn_init(T *wn_r, T *wn_i, const int N)
{
    T *wreal = (T *)malloc(sizeof(T) * N);
    T *wimag = (T *)malloc(sizeof(T) * N);
    T arg = - 2 * PI / N;
    T treal = cos(arg);
    T timag = sin(arg);
    wreal[0] = 1.0;
    wimag[0] = 0.0;
    for(int j = 1; j < N / 2; ++j)
    {
        wreal[j] = wreal[j - 1] * treal - wimag[j - 1] * timag;
        wimag[j] = wreal[j - 1] * timag + wimag[j - 1] * treal;
    }
    cudaMemcpy(wn_r, wreal, sizeof(T) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(wn_i, wimag, sizeof(T) * N, cudaMemcpyHostToDevice);
    free(wreal);
    free(wimag);
}

template <typename T>
__global__
void fft_kernel(const T *wn_r, const T *wn_i, T *cr, T *ci, const int N)
{
    __shared__ T sr[2048];
    __shared__ T si[2048];

    int p = 0;
    for(int i = 1; i < N; i <<= 1)
        p++;

    for(int tid = threadIdx.x; tid < N; tid += blockDim.x)
    {
        int a = tid;
        int b = 0;
        for(int j = 0; j < p; j++)
        {
            b = (b << 1) + (a & 1);    // b = b * 2 + a % 2;
            a >>= 1;        // a = a / 2;
        }
        sr[tid] = cr[b];
        si[tid] = ci[b];
    }

    __syncthreads();

    for(int k = 0, mod = 1, stride = 2; k < p; ++k, mod <<= 1, stride <<= 1)
    {
        for(int tid = threadIdx.x; tid < N; tid += blockDim.x)
        {
            int id1 = tid;
            int id2 = id1 ^ (1 << k);

            if(id1 < id2)
            {
                int tidx = N * (id1 % mod) / stride;
                T tr = __ldg(&wn_r[tidx]);
                T ti = __ldg(&wn_i[tidx]);

                T nr = tr * sr[id2] - ti * si[id2];
                T ni = tr * si[id2] + ti * sr[id2];

                //pm = pi - t
                sr[id2] = sr[id1] - nr;
                si[id2] = si[id1] - ni;
                //pi += t
                sr[id1] += nr;
                si[id1] += ni;
                //printf("k %d i %d id1 %d id2 %d t(%lf, %lf) val1 (%lf, %lf) val2 (%lf, %lf)\n", k, id1 % mod, id1, id2,
                //      tr, ti,
                //       sr[id1], si[id1], sr[id2], si[id2]);
            }
        }
        __syncthreads();
    }
    for(int tid = threadIdx.x; tid < N; tid += blockDim.x)
    {
        cr[tid] = sr[tid];
        ci[tid] = si[tid];

    }
}

template <typename T>
__global__
void fft_kernel2(const T *wn_r, const T *wn_i, T *cr, T *ci, const int N)
{
    __shared__ T sr[2048];
    __shared__ T si[2048];

    int p = 0;
    for(int i = 1; i < N; i <<= 1)
        p++;

    for(int tid = threadIdx.x; tid < N; tid += blockDim.x)
    {
        int a = tid;
        int b = 0;
        for(int j = 0; j < p; j++)
        {
            b = (b << 1) + (a & 1);    // b = b * 2 + a % 2;
            a >>= 1;        // a = a / 2;
        }
        sr[tid] = cr[b];
        si[tid] = ci[b];
        //printf("tid %d b %d val (%lf, %lf)\n", tid, b, sr[tid], si[tid]);
    }
    __syncthreads();

    int half_N = N / 2;
    //int half_N = 4;
    for(int stride = 1; stride <= half_N; stride <<= 1)
    {
        for(int tid = threadIdx.x; tid < half_N; tid += blockDim.x)
        {
            int id1 = (tid % stride) + (tid / stride) * (stride << 1);
            int id2 = id1 + stride;

            int tidx = N * (id1 % stride) / (stride << 1);

            T tr = __ldg(&wn_r[tidx]);
            T ti = __ldg(&wn_i[tidx]);

            T nr = tr * sr[id2] - ti * si[id2];
            T ni = tr * si[id2] + ti * sr[id2];


            //pm = pi - t
            sr[id2] = sr[id1] - nr;
            si[id2] = si[id1] - ni;
            //pi += t
            sr[id1] += nr;
            si[id1] += ni;
            //printf("tid %d tidx %d id1 %d id2 %d\n", tid, tidx, id1, id2);
        }
        if(stride >= 32)
            __syncthreads();
    }
    for(int tid = threadIdx.x; tid < N; tid += blockDim.x)
    {
        cr[tid] = sr[tid];
        ci[tid] = si[tid];
    }
}

template <typename T>
__global__
void fft_kernel_round1(const T *wn_r, const T *wn_i, T *cr, T *ci, const int N)
{
    __shared__ T sr[2048];
    __shared__ T si[2048];

    int p = 0;

    int each_block_N = N / gridDim.x;
    int block_offset = each_block_N * blockIdx.x;
    for(int i = 1; i < N; i <<= 1)
        p++;

    for(int tid = threadIdx.x; tid < each_block_N; tid += blockDim.x)
    {
        int a = tid + block_offset;
        int b = 0;
        for(int j = 0; j < p; j++)
        {
            b = (b << 1) + (a & 1);    // b = b * 2 + a % 2;
            a >>= 1;        // a = a / 2;
        }
        sr[tid] = cr[b];
        si[tid] = ci[b];
    }
    __syncthreads();

    int half_N = each_block_N / 2;
    for(int stride = 1; stride <= half_N; stride <<= 1)
    {
        for(int tid = threadIdx.x; tid < half_N; tid += blockDim.x)
        {
            int id1 = (tid % stride) + (tid / stride) * (stride << 1);
            int id2 = id1 + stride;

            int tidx = N * (id1 % stride) / (stride << 1);

            T tr = __ldg(&wn_r[tidx]);
            T ti = __ldg(&wn_i[tidx]);

            T nr = tr * sr[id2] - ti * si[id2];
            T ni = tr * si[id2] + ti * sr[id2];


            //pm = pi - t
            sr[id2] = sr[id1] - nr;
            si[id2] = si[id1] - ni;
            //pi += t
            sr[id1] += nr;
            si[id1] += ni;
            //printf("tid %d tidx %d id1 %d id2 %d\n", tid, tidx, id1, id2);
        }
        if(stride >= 32)
            __syncthreads();
    }
    for(int tid = threadIdx.x; tid < each_block_N; tid += blockDim.x)
    {
        cr[tid + block_offset] = sr[tid];
        ci[tid + block_offset] = si[tid];
    }
}

template <typename T>
__global__
void fft_kernel_round2(const T *wn_r, const T *wn_i, T *cr, T *ci, const int N, const int first_p)
{
    int half_N = N / 2;
    for(int stride = (1 << first_p); stride <= half_N; stride <<= 1)
    {
        for(int tid = threadIdx.x; tid < half_N; tid += blockDim.x)
        {
            int id1 = (tid % stride) + (tid / stride) * (stride << 1);
            int id2 = id1 + stride;

            int tidx = N * (id1 % stride) / (stride << 1);

            T tr = __ldg(&wn_r[tidx]);
            T ti = __ldg(&wn_i[tidx]);

            T cr2_val = cr[id2];
            T ci2_val = ci[id2];
            T cr1_val = cr[id1];
            T ci1_val = ci[id1];

            T nr = tr * cr2_val - ti * ci2_val;
            T ni = tr * ci2_val + ti * cr2_val;


            //pm = pi - t
            cr[id2] = cr1_val - nr;
            ci[id2] = ci1_val - ni;
            //pi += t
            cr[id1] = cr1_val + nr;
            ci[id1] = ci1_val + ni;
            //printf("tid %d tidx %d id1 %d id2 %d\n", tid, tidx, id1, id2);
        }
        __syncthreads();
    }
}

template <typename T>
void fft_kernel_launcher(const T *wn_r, const T *wn_i, T *cr, T *ci, const int N)
{
    dim3 grid(1);
    dim3 block(512);
    //fft_kernel<<<grid, block>>>(wn_r, wn_i, cr, ci, N);
    //fft_kernel2<<<grid, block>>>(wn_r, wn_i, cr, ci, N);

    block.x = 64;
    grid.x = N / block.x / 2;
    fft_kernel_round1<<<grid, block>>>(wn_r, wn_i, cr, ci, N);

    int first_p = 0;
    for(int i = 1; i < block.x * 2; i *= 2)
        first_p++;

    grid.x = 1;
    block.x = 1024;
    fft_kernel_round2<<<grid, block>>>(wn_r, wn_i, cr, ci, N, first_p);
}

template <typename T>
__global__
void data_vreff_dot_kernel(const double *vr_eff, T *data, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;
    for(; tid < N; tid += total_threads)
    {
        T data_val = data[tid];
        double vr_eff_val = __ldg(&vr_eff[tid]);
        data_val.x *= vr_eff_val;
        data_val.y *= vr_eff_val;
        data[tid] = data_val;
    }
}

template <typename T>
__global__
void data_scale_kernel(T *data, const int N, const double scaler)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;
    for(; tid < N; tid += total_threads)
    {
        T data_val = data[tid];
        data_val.x *= scaler;
        data_val.y *= scaler;
        data[tid] = data_val;
    }
}


template <typename T>
void FFT<T>::fft3d_all(const double *vr_eff, T *data, cudaStream_t stream)
{
    cufftSetStream(plan_, stream);
    cufftExecZ2Z(plan_, (cufftDoubleComplex *)data, (cufftDoubleComplex *)data, CUFFT_INVERSE);

    dim3 grid;
    dim3 block(512);
    grid.x = (N_ - 1) / block.x + 1;
    data_vreff_dot_kernel<<<grid, block, 0, stream>>>(vr_eff, data, N_);

    cufftExecZ2Z(plan_, (cufftDoubleComplex *)data, (cufftDoubleComplex *)data, CUFFT_FORWARD);

    double scaler = 1.0 / (nx_ * ny_ * nz_);
    data_scale_kernel<<<grid, block, 0, stream>>>(data, N_, scaler);
}

template void wn_init(double *wn_r, double *wn_i, const int N);
template void fft_kernel_launcher(const double *wn_r, const double *wn_i, double *cr, double *ci, const int N);
template void FFT<cuDoubleComplex>::fft3d_all(const double *vreff, cuDoubleComplex *data, cudaStream_t stream);

}

