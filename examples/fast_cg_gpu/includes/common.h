#pragma once
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <cufftXt.h>
enum class OperationType {FP64, FP32};

template<OperationType OpType>
class Traits;

template<>
class Traits<OperationType::FP64>
{
public:
    typedef double ElemType;
    typedef cuDoubleComplex ComplexType;
};

template<>
class Traits<OperationType::FP32>
{
public:
    typedef float ElemType;
    typedef cuComplex ComplexType;
};


#define PRINT_FUNC_NAME_() do{\
        std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl; \
    } while (0)

static const char *_cudaGetErrorEnum(cudaError_t error)
{
    return cudaGetErrorString(error);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch(error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file, int const line)
{
    if(result)
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + \
                                 (_cudaGetErrorEnum(result)) + " " + file +  \
                                 ":" + std::to_string(line) + " \n");
}

#define CUTLASS_CHECK(status)                                                                    \
    {                                                                                              \
        cutlass::Status error = status;                                                              \
        if (error != cutlass::Status::kSuccess) {                                                    \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                      << std::endl;                                                                    \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                            \
    }

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
