#include "../includes/orth.h"
#include "../includes/common.h"
#include "../includes/complex_utils.h"
#include "helper.h"
#include "utils.h"
#include "complex.h"
#include <cufft.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigenvalues>

using Eigen::SelfAdjointEigenSolver;

typedef Eigen::MatrixXd MatrixType;
typedef Eigen::VectorXd VectorType;
typedef Eigen::VectorXcd ComplexVectorType;
typedef Eigen::MatrixXcd ComplexMatrixType;
typedef double T;
typedef std::complex<double> ComplexT;

using namespace std;
using namespace faster_dft;


template <OperationType OpType>
void Test(const int M, const int K, const int N)
{
    typedef Traits<OpType> Traits;
    typedef typename Traits::ComplexType ComplexType;
    typedef typename Traits::ElemType ElemDataType;

    ComplexMatrixType A = randomMatrix(M, K);
    ComplexMatrixType B = randomMatrix(K, N);
    ComplexMatrixType C = randomMatrix(M, N);

    ComplexType *d_A;
    ComplexType *d_B;
    ComplexType *d_C;

    cudaMalloc((void **)&d_A, sizeof(ComplexType) * M * K);
    cudaMalloc((void **)&d_B, sizeof(ComplexType) * K * N);
    cudaMalloc((void **)&d_C, sizeof(ComplexType) * M * N);

    init_vector(d_A, A);
    init_vector(d_B, B);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    complex_gemm(d_A, d_B, d_C, M, K, N, handle);


    if(OpType == OperationType::FP64)
        printf("\n********************\nGEMM benchmark FP64 [%d, %d, %d] Testing\n", M, K, N);
    if(OpType == OperationType::FP32)
        printf("\n********************\nGEMM benchmark FP32 [%d, %d, %d] Testing\n", M, K, N);

    struct timeval ss, ee;
    int ite = 10;
    cudaDeviceSynchronize();
    gettimeofday(&ss, NULL);
    for(int i = 0; i < ite; ++i)
    {
        complex_gemm(d_A, d_B, d_C, M, K, N, handle);
    }
    cudaDeviceSynchronize();
    gettimeofday(&ee, NULL);
    printf("gemm costs %.2f ms\n", diffTime(ss, ee) / ite);

    C = A * B;
    check_diff(d_C, C);

}

int main(int argc, char *argv[])
{
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    Test<OperationType::FP64>(M, K, N);
    Test<OperationType::FP32>(M, K, N);
}
