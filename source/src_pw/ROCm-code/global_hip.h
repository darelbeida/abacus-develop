#ifndef GLOBAL_H
#define GLOBAL_H

#include "../run_pw.h"
#include "../module_base/global_variable.h"
#include "../module_base/global_function.h"
#include "pw_basis.h"
#include "energy.h"
#include "VNL_in_pw.h"
#include "charge_broyden.h"
#include "potential.h"
#include "xc_type.h"
#include "hamilt.h"
#include "../src_ions/ions.h"
#include "wavefunc.h"
#include "use_fft.h"
#include "klist.h"
#include "../src_io/output.h"
#include "magnetism.h"
#include "vdwd2.h"
#include "vdwd2_parameters.h"
#include "vdwd3.h"
#include "vdwd3_parameters.h"	
#include "../src_io/restart.h" 
#include "exx_global.h"
#include "../src_lcao/exx_lip.h"
#include "../src_parallel/ft.h"

// TODO: add new macro __ROCm ...
#ifdef __CUDA
static const char *_hipblasGetErrorEnum(hipblasStatus_t error)
{
    switch (error)
    {
        case HIPBLAS_STATUS_SUCCESS:
            return "HIPBLAS_STATUS_SUCCESS";
        case HIPBLAS_STATUS_NOT_INITIALIZED:
            return "HIPBLAS_STATUS_NOT_INITIALIZED";
        case HIPBLAS_STATUS_ALLOC_FAILED:
            return "HIPBLAS_STATUS_ALLOC_FAILED";
        case HIPBLAS_STATUS_INVALID_VALUE:
            return "HIPBLAS_STATUS_INVALID_VALUE";
        case HIPBLAS_STATUS_ARCH_MISMATCH:
            return "HIPBLAS_STATUS_ARCH_MISMATCH";
        case HIPBLAS_STATUS_MAPPING_ERROR:
            return "HIPBLAS_STATUS_MAPPING_ERROR";
        case HIPBLAS_STATUS_EXECUTION_FAILED:
            return "HIPBLAS_STATUS_EXECUTION_FAILED";
        case HIPBLAS_STATUS_INTERNAL_ERROR:
            return "HIPBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}

// rocSolver or hipSolver
static const char *_cusolverGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}

static const char *_hipfftGetErrorEnum(hipfftResult_t error)
{
    switch (error)
    {
        case HIPFFT_SUCCESS:
            return "HIPFFT_SUCCESS";
        case HIPFFT_INVALID_PLAN:
            return "HIPFFT_INVALID_PLAN";
        case HIPFFT_ALLOC_FAILED:
            return "HIPFFT_ALLOC_FAILED";
        case HIPFFT_INVALID_TYPE:
            return "HIPFFT_INVALID_TYPE";
        case HIPFFT_INVALID_VALUE:
            return "HIPFFT_INVALID_VALUE";
        case HIPFFT_INTERNAL_ERROR:
            return "HIPFFT_INTERNAL_ERROR";
        case HIPFFT_EXEC_FAILED:
            return "HIPFFT_EXEC_FAILED";
        case HIPFFT_SETUP_FAILED:
            return "HIPFFT_SETUP_FAILED";
        case HIPFFT_INVALID_SIZE:
            return "HIPFFT_INVALID_SIZE";
        case HIPFFT_UNALIGNED_DATA:
            return "HIPFFT_UNALIGNED_DATA";
        case HIPFFT_INCOMPLETE_PARAMETER_LIST:
            return "HIPFFT_INCOMPLETE_PARAMETER_LIST";
        case HIPFFT_INVALID_DEVICE:
            return "HIPFFT_INVALID_DEVICE";
        case HIPFFT_PARSE_ERROR:
            return "HIPFFT_PARSE_ERROR";
        case HIPFFT_NO_WORKSPACE:
            return "HIPFFT_NO_WORKSPACE";
        case HIPFFT_NOT_IMPLEMENTED:
            return "HIPFFT_NOT_IMPLEMENTED";
        case CUFFT_LICENSE_ERROR:
            return "CUFFT_LICENSE_ERROR";
        case HIPFFT_NOT_SUPPORTED:
            return "HIPFFT_NOT_SUPPORTED";
    }
    return "<unknown>";
}

#define CHECK_CUDA(func)\
{\
    hipError_t status = (func);\
    if(status != hipSuccess)\
    {\
        printf("In File %s : HIP API failed at line %d with error: %s (%d)\n",\
            __FILE__, __LINE__, hipGetErrorString(status), status);\
    }\
}

#define CHECK_CUBLAS(func)\
{\
    hipblasStatus_t status = (func);\
    if(status != HIPBLAS_STATUS_SUCCESS)\
    {\
        printf("In File %s : HIPBLAS API failed at line %d with error: %s (%d)\n",\
            __FILE__, __LINE__, _hipblasGetErrorEnum(status), status);\
    }\
}

#define CHECK_CUSOLVER(func)\
{\
    cusolverStatus_t status = (func);\
    if(status != CUSOLVER_STATUS_SUCCESS)\
    {\
        printf("In File %s : CUSOLVER API failed at line %d with error: %s (%d)\n",\
            __FILE__, __LINE__, _cusolverGetErrorEnum(status), status);\
    }\
}

#define CHECK_CUFFT(func)\
{\
    hipfftResult_t status = (func);\
    if(status != HIPFFT_SUCCESS)\
    {\
        printf("In File %s : HIPFFT API failed at line %d with error: %s (%d)\n",\
            __FILE__, __LINE__, _hipfftGetErrorEnum(status), status);\
    }\
}
#endif


//==========================================================
// EXPLAIN : define "GLOBAL CLASS"
//==========================================================
namespace GlobalC
{
extern K_Vectors kv;
extern Use_FFT UFFT;
extern output out;
extern PW_Basis pw;
extern Stochastic_WF sto_wf; //qianrui add 2021-2-5
extern energy en;
extern wavefunc wf;
extern Hamilt hm;
extern Exx_Global exx_global;
extern Exx_Lip exx_lip;
extern pseudopot_cell_vnl ppcell;
}


#include "../module_symmetry/symmetry.h"
#include "../module_cell/unitcell_pseudo.h"
#include "../src_parallel/parallel_grid.h"
#include "../src_parallel/parallel_kpoints.h"
namespace GlobalC
{
extern UnitCell_pseudo ucell;
extern xcfunc xcf;
extern Charge_Broyden CHR;
extern Potential pot;
extern ModuleSymmetry::Symmetry symm;
extern Parallel_Grid Pgrid; 
extern Parallel_Kpoints Pkpoints;
extern Vdwd2_Parameters vdwd2_para;		// Peize Lin add 2021.03.09
extern Vdwd3_Parameters vdwd3_para;		// jiyy add 2021-05-02	
extern Restart restart;	// Peize Lin add 2020.04.04
}

//extern Magnetism mag;

#ifdef __LCAO
#include "../src_lcao/global_fp.h"
#endif

#endif