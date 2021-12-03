#include "../includes/complex_utils.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cublas_v2.h>

namespace faster_dft
{

template <>
void complex_gemm(const cuDoubleComplex *A, const cuDoubleComplex *B, cuDoubleComplex *C,
                  const int M, const int K, const int N, cublasHandle_t handle)
{
    cuDoubleComplex ONE, ZERO;
    ONE.y = ZERO.x = ZERO.y = 0.0;
    ONE.x = 1.0;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    cublasZgemm(handle, transa, transb, N, M, K, &ONE, B, N, A, K, &ZERO, C, N);

}

template <>
void complex_gemm(const cuComplex *A, const cuComplex *B, cuComplex *C,
                  const int M, const int K, const int N, cublasHandle_t handle)
{
    cuComplex ONE, ZERO;
    ONE.y = ZERO.x = ZERO.y = 0.0;
    ONE.x = 1.0;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    cublasCgemm(handle, transa, transb, N, M, K, &ONE, B, N, A, K, &ZERO, C, N);

}

template void complex_gemm(const cuDoubleComplex *A, const cuDoubleComplex *B, cuDoubleComplex *C, const int M, const int K, const int N, cublasHandle_t handle);
template void complex_gemm(const cuComplex *A, const cuComplex *B, cuComplex *C, const int M, const int K, const int N, cublasHandle_t handle);
}
