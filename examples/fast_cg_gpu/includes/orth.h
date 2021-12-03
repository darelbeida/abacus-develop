#pragma once
//#include <cublas_v2.h>
#include "complex_utils.h"
#include "common.h"

namespace faster_dft
{

template <OperationType OpType>
struct OrthParam
{
    typedef Traits<OpType> Traits_;
    typedef typename Traits_::ComplexType T;
    T *X;
    T *V;
    void *buf;
    int K;
    const int N;
    bool norm;
    cublasHandle_t cublas_handle;
    cudaStream_t stream;
};


template <OperationType OpType>
void orth_kernel_launcher(OrthParam<OpType> param);

}
