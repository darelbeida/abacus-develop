/*
* Author: Xiaoying Jia
* Project: DFT
* Department: ByteDance Data-AML
* Email: {jiaxiaoying}@bytedance.com
*/
#include "../includes/orth.h"
#include <cooperative_groups.h>
#include <cmath>
using namespace std;
using namespace cooperative_groups;
namespace cg = cooperative_groups;

#define WARP_SIZE  32

namespace faster_dft
{
#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T val)
{
    for(int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if(lane == 0)
        shared[wid] = val;
    __syncthreads();

    return wid == 0 ? warpReduceSum(threadIdx.x < (blockDim.x >> 5) ? shared[lane] : (T)0.0) : 0.0;
}

/*

template <typename T>
__global__
void orth_kernel(const T *X_ptr, T *V_ptr, T *global_dot_val, const int K, const int N, const bool norm)
{
    cg::grid_group g = cg::this_grid();
    int global_id = threadIdx.x + blockIdx.x * blockDim.x;

    T v = (global_id < N) ? V_ptr[global_id] : 0.0;

    for(int ite = 0; ite < K; ++ite)
    {
        T x = (global_id < N) ? X_ptr[global_id + ite * N] : 0.0;
        T val = x * v;
        T dot_val = blockReduceSum(val);
        if(threadIdx.x == 0)
            atomicAdd(&global_dot_val[ite], dot_val);
        cg::sync(g);

        if(global_id < N)
            v -= global_dot_val[ite] * x;
    }
    if(global_id < N)
        V_ptr[global_id] = v;
}

template <typename T>
void orth_kernel_launcher(const T *X, T *V, T *dot_val, const int K, const int N, bool norm)
{
    dim3 block(128);
    dim3 grid((N - 1)/ block.x + 1);
    int sMemSize = sizeof(T) * 32;

    cudaMemset(&dot_val, 0, sizeof(T) * K);
    void *kernelArgs[] = {(void *) &X, (void *) &V, (void *) &dot_val, (void *) &K, (void *) &N, (void *) &norm};
    cudaLaunchCooperativeKernel((void *)orth_kernel<T>, grid, block, kernelArgs, sMemSize, NULL);
}
*/



template void orth_kernel_launcher(const cuDoubleComplex *X, cuDoubleComplex *V, void* buf, const int K, const int N, const bool norm);

}

