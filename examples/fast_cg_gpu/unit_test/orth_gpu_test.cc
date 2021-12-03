#include "../includes/orth.h"
#include "../includes/common.h"
#include "helper.h"
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

// 用施密特正交化让v和X的前K个列向量正交
void orth(const ComplexMatrixType &X, const int K, ComplexVectorType &v, bool norm = true)
{
    if(K == 0)
    {
        if(norm) v /= v.norm();
    }
    else
    {
        ComplexVectorType xv = X(Eigen::all, Eigen::seqN(0, K)).adjoint() * v;
        v -= X(Eigen::all, Eigen::seqN(0, K)) * xv;
    }
}

ComplexMatrixType randomSymmetryMatrix(int N)
{
    ComplexMatrixType A = ComplexMatrixType::Zero(N, N);
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            T e1 = sin(i * 0.23 + 0.35 * j) + 0.017 * ((i * j) % 100);
            T e2 = sin(i * 0.71 + 0.11 * j) + 0.013 * ((i * j) % 100);
            A(i, j) = std::complex<T>(e1, e2);
        }
    }
    return A;
}


template <OperationType OpType>
void Test()
{
    typedef Traits<OpType> Traits;
    typedef typename Traits::ComplexType ComplexType;
    typedef typename Traits::ElemType ElemDataType;

    const int N = 9937;
    const int K = 130;

    ComplexMatrixType A = randomSymmetryMatrix(N);
    ComplexVectorType v = A.col(0);

    ComplexType *h_A = (ComplexType *)malloc(sizeof(ComplexType) * N * N);
    ComplexType *h_v = (ComplexType *)malloc(sizeof(ComplexType) * N);
    ComplexType *d_A;
    ComplexType *d_v;
    ComplexType *d_v1;
    ComplexType *d_xv;

    cudaMalloc((void **)&d_A, sizeof(ComplexType) * N * N);
    cudaMalloc((void **)&d_v, sizeof(ComplexType) * N);
    cudaMalloc((void **)&d_v1, sizeof(ComplexType) * N);
    cudaMalloc((void **)&d_xv, sizeof(ComplexType) * K);

    //h_A = A.transpose()
    for(int i = 0; i < N; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            h_A[i * N + j].x = A(j, i).real();
            h_A[i * N + j].y = A(j, i).imag();
        }
    }

    for(int i = 0; i < N; ++i)
    {
        h_v[i].x = v(i).real();
        h_v[i].y = v(i).imag();
    }

    cudaMemcpy(d_A, h_A, sizeof(ComplexType) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, sizeof(ComplexType) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v1, h_v, sizeof(ComplexType) * N, cudaMemcpyHostToDevice);

    cublasHandle_t diag_handle;
    cublasCreate(&diag_handle);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(diag_handle, stream);


    OrthParam<OpType> param{d_A, d_v1, (void *)d_xv, K, N, false, diag_handle, stream};
    orth_kernel_launcher<OpType>(param);
    cudaMemcpy(h_v, d_v1, sizeof(ComplexType) * N, cudaMemcpyDeviceToHost);

    if(OpType == OperationType::FP64)
        printf("\n********************\nOrth FP64 Testing\n");
    if(OpType == OperationType::FP32)
        printf("\n********************\nOrth FP32 Testing\n");

    struct timeval ss, ee;
    int ite = 10;
    for(int kk = K; kk <= K; kk++)
    {
        OrthParam<OpType> param{d_A, d_v1, (void *)d_xv, kk, N, false, diag_handle, stream};
        cudaDeviceSynchronize();
        gettimeofday(&ss, NULL);
        for(int i = 0; i < ite; ++i)
        {
            orth_kernel_launcher(param);
            //cublasZgemv(diag_handle, trans1, N, kk, &ONE, d_A, N, d_v, 1, &ZERO, d_xv, 1);
            //cublasZgemv(diag_handle, trans2, N, kk, &NEG_ONE, d_A, N, d_xv, 1, &ONE, d_v, 1);
        }
        cudaDeviceSynchronize();
        gettimeofday(&ee, NULL);
        printf("orth N %d K %d costs %.2f ms\n", N, kk, diffTime(ss, ee) / ite);
    }

    orth(A, K, v, false);

    for(int i = 0; i < min(10, K); ++i)
        printf("i %d (%lf, %lf) gpu (%lf, %lf) \n", i, v[i].real(), v[i].imag(), h_v[i].x, h_v[i].y);
}

int main()
{
    Test<OperationType::FP64>();
    Test<OperationType::FP32>();
}
