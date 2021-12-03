/*
* Author: Xiaoying Jia
* Project: FasterDFT
* Department: ByteDance Data-AML
* Email: {jiaxiaoying}@bytedance.com
*/
#include "../includes/orth.h"
#include "../includes/complex_utils.h"
#include "../includes/reduce.h"
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
__global__
void orth_1st_round_kernel(const T *X_ptr, const T *V_ptr, T *xv_ptr, const int K, const int N, const int each_N_block)
{
    int each_blocks = gridDim.x / K;
    int bid = blockIdx.x / each_blocks;
    int offset = (blockIdx.x % each_blocks) * each_N_block;

    T xv;
    xv.x = 0.0;
    xv.y = 0.0;

    for(int tid = threadIdx.x + offset; tid < min(offset + each_N_block, N); tid += blockDim.x)
    {
        T x = __ldg(&X_ptr[bid * N + tid]);
        T v = __ldg(&V_ptr[tid]);
        T dot;
        dot.x = x.x * v.x + x.y * v.y;
        dot.y = x.x * v.y - x.y * v.x;

        xv.x += dot.x;
        xv.y += dot.y;
    }

    xv.x = blockReduceSum(xv.x);
    xv.y = blockReduceSum(xv.y);
    if(threadIdx.x == 0)
    {
        atomicAdd(&xv_ptr[bid].x, xv.x);
        atomicAdd(&xv_ptr[bid].y, xv.y);
    }
}


template <typename T>
__global__
void orth_2nd_round_kernel(const T *X_ptr, const T *xv_ptr, T *v_ptr, const int K, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= N)
        return;

    T xxv;
    xxv.x = 0.0;
    xxv.y = 0.0;

    for(int i = 0; i < K; ++i)
    {
        T x = __ldg(&X_ptr[i * N + tid]);
        T xv = __ldg(&xv_ptr[i]);

        T dot;
        dot.x = x.x * xv.x - x.y * xv.y;
        dot.y = x.x * xv.y + x.y * xv.x;

        xxv.x += dot.x;
        xxv.y += dot.y;
    }
    v_ptr[tid].x = v_ptr[tid].x - xxv.x;
    v_ptr[tid].y = v_ptr[tid].y - xxv.y;
}

template <typename ComplexType, typename ElemType>
__global__
void vector_norm_kernel(ComplexType *V, ElemType *norm_val, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ ElemType norm_scaler;
    ComplexType val;
    if(tid < N)
    {
        val = V[tid];
    }
    else
    {
        val.x = 0.0;
        val.y = 0.0;
    }

    ElemType sqrt_sum = val.x * val.x + val.y * val.y;

    sqrt_sum = blockReduceSum(sqrt_sum);

    cg::grid_group g = cg::this_grid();
    if(threadIdx.x == 0)
    {
        atomicAdd(&norm_val[0], sqrt_sum);
    }
    cg::sync(g);
    if(threadIdx.x == 0)
    {
        norm_scaler = rsqrt(norm_val[0]);
    }
    __syncthreads();
    if(tid < N)
    {
        V[tid].x = val.x * norm_scaler;
        V[tid].y = val.y * norm_scaler;
    }
}

template <typename ComplexType, typename ElemType>
void norm_kernel_launcher(ComplexType *V, ElemType *norm_val, const int N, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((N - 1)/ block.x + 1);

    cudaMemsetAsync(norm_val, 0, sizeof(ElemType), stream);
    int sMemSize = 0;
    void *kernelArgs[] = {(void *) &V, (void *) &norm_val, (void *) &N};
    cudaLaunchCooperativeKernel((void *)vector_norm_kernel<ComplexType, ElemType>, grid, block, kernelArgs, sMemSize, stream);
}
template <OperationType OpType>
void orth_kernel_launcher(OrthParam<OpType> param)
{
    typedef Traits<OpType> Traits;
    typedef typename Traits::ComplexType ComplexType;
    typedef typename Traits::ElemType ElemType;

    const int N = param.N;
    const int K = param.K;
    const ComplexType *X = param.X;
    ComplexType *V = param.V;
    ComplexType *xv = (ComplexType *)(param.buf);
    ElemType *norm_val = (ElemType *)(xv + K);

    /*
    printf("orth params N %d K %d\n", N, K);
    print_complex(X, 10, "orth X");
    print_complex(V, 10, "orth V");
    */

    if(K == 0)
    {
        if(!param.norm)
            return;
        norm_kernel_launcher<ComplexType, ElemType>(V, norm_val, N, param.stream);
        return;
    }

    int each_block_N = 2048;
    dim3 block(256);
    dim3 grid(K * ((N - 1) / each_block_N + 1));

    cudaMemsetAsync(xv, 0, sizeof(ComplexType) * K, param.stream);
    orth_1st_round_kernel<<<grid, block, 0, param.stream>>>(X, V, xv, K, N, each_block_N);

    cublasOperation_t trans2 = CUBLAS_OP_N;

    //x: K x N
    //vx: 1 x K
    if(OpType == OperationType::FP64)
    {
        cuDoubleComplex ONE, ZERO, NEG_ONE;
        ONE.y = ZERO.x = ZERO.y = 0.0;
        ONE.x = 1.0;
        NEG_ONE.x = -1.0;
        cublasZgemv(param.cublas_handle, trans2, N, K, &NEG_ONE, (const cuDoubleComplex *) X, N, (const cuDoubleComplex *)xv, 1,
                    &ONE, (cuDoubleComplex *)V, 1);
    }
    else
    {
        cuComplex ONE, ZERO, NEG_ONE;
        ONE.y = ZERO.x = ZERO.y = 0.0f;
        ONE.x = 1.0f;
        NEG_ONE.x = -1.0f;
        cublasCgemv(param.cublas_handle, trans2, N, K, &NEG_ONE, (const cuComplex *) X, N, (const cuComplex *)xv, 1,
                    &ONE, (cuComplex *)V, 1);
    }

    if(param.norm)
        norm_kernel_launcher<ComplexType, ElemType>(V, norm_val, N, param.stream);

    /*

    block.x = 64;
    grid.x = (N - 1) / block.x + 1;
    orth_2nd_round_kernel<<<grid, block>>>(X, xv, V, K, N);
    */
}



template void orth_kernel_launcher<OperationType::FP64>(OrthParam<OperationType::FP64> param);
template void orth_kernel_launcher<OperationType::FP32>(OrthParam<OperationType::FP32> param);

}

