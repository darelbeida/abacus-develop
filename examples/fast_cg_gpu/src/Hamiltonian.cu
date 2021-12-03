/*
* Author: Xiaoying Jia
* Project: FasterDFT
* Department: ByteDance Data-AML
* Email: {jiaxiaoying}@bytedance.com
*/
#include "../includes/Hamiltonian.h"
#include "../includes/complex_utils.h"
#include "../includes/reduce.h"
//#include <cublas_v2.h>
//#include <cufft.h>
#include <cmath>
using namespace std;

#define WARP_SIZE  32
#define PI 3.141592653589793

namespace faster_dft
{

template <typename T>
__global__
void init_fft_data_kernel(const T *in, T *out, const int *index, const int N, const int Nxyz, const int K)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;
    for(; tid < N; tid += total_threads)
    {
		int index_id = index[tid];
		for(int idx = 0; idx < K; ++idx)
		{
			out[idx * Nxyz + index_id] = __ldg(&in[idx * N + tid]);
		}
    }
}

template <typename T>
__global__
void cal_becp_kernel(const T *vkb_conjugate, const T *x, T *becp, const int rows, const int cols)
{
    int row_id = blockIdx.x;

    T val;
    val.x = 0.0;
    val.y = 0.0;
#pragma unroll
    for(int tid = threadIdx.x; tid < cols; tid += blockDim.x)
    {
        T vkb_val = __ldg(&vkb_conjugate[row_id * cols + tid]);
        T x_val = __ldg(&x[tid]);
        val.x += vkb_val.x * x_val.x - vkb_val.y * x_val.y;
        val.y += vkb_val.x * x_val.y + vkb_val.y * x_val.x;
    }
    val.x = blockReduceSum(val.x);
    val.y = blockReduceSum(val.y);
    if(threadIdx.x == 0)
    {
        becp[row_id] = val;
    }
}

template <typename ComplexType, typename ElemType>
__global__
void cal_ps_kernel(const ElemType *deeq, const ComplexType *becp, ComplexType *ps, const int batch, const int N)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;

    ComplexType val;
    val.x = 0.0;
    val.y = 0.0;

    //8x8 x 8x1 -> 8
    for(int i = 0; i < N; ++i)
    {
        ElemType scale = __ldg(&deeq[bid * N * N + tid * N + i]);
        ComplexType becp_val = __ldg(&becp[bid * N + i]);
        val.x += becp_val.x * scale;
        val.y += becp_val.y * scale;
    }

    ps[bid * N + tid] = val;
}

template <typename ComplexType, typename ElemType>
__global__
void update_y_kernel(const ComplexType *x, const ElemType *g2_kin, const ComplexType *fft_data,
                     const int *gr_index, const ComplexType *vkb_ps, ComplexType *hx, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= N)
        return;

    hx[tid].x = 0.0;
    hx[tid].y = 0.0;

    hx[tid] = fft_data[gr_index[tid]];
    hx[tid].x += vkb_ps[tid].x;
    hx[tid].y += vkb_ps[tid].y;

    hx[tid].x += __ldg(&x[tid]).x * g2_kin[tid];
    hx[tid].y += __ldg(&x[tid]).y * g2_kin[tid];
}
template <OperationType OpType>
void Hamiltonian<OpType>::apply(const ComplexType *x, ComplexType *hx, void *buf, cublasHandle_t handle, cudaStream_t stream)
{
    //printf("Hamiltonain apply N_ %d\n", N_);
    cudaMemsetAsync(fft_data, 0, sizeof(ComplexType) * n_xyz_ * K_, stream);
    dim3 grid;
    dim3 block(128);
    grid.x = (N_ - 1) / block.x + 1;

    init_fft_data_kernel<<<grid, block, 0, stream>>>(x, fft_data, param_.gr_index, N_, n_xyz_, K_);

    fft3d_->fft3d_all(param_.vr_eff, fft_data, stream);
	
	printf("after fft \n");
	for(int i = 0; i < K_; ++i)
	{
		cout << "i " << i << endl;
		print_complex(fft_data + i * n_xyz_, 10, "fft_data");
	}

    cublasOperation_t trans;


    //224 x 9937 matmul 9937 -> 224
    grid.x = natom_ * nproj_;
    block.x = 1024;

    //cal_becp_kernel<<<grid, block, 0, stream>>>(param_.vkb_conjugate, x, becp, natom_ * nproj_, N_);

	cuDoubleComplex ONE, ZERO;
    ONE.y = ZERO.x = ZERO.y = 0.0;
    ONE.x = 1.0;



	//X [K, N]
	//vkb_conjudate [natom_ * nproj_, N]
	//becp [K, natom_ * nproj_]
    if(OpType == OperationType::FP64)
	{
		cublasOperation_t transa = CUBLAS_OP_T;
		cublasOperation_t transb = CUBLAS_OP_N;
		const int N = K_;
		const int K = N_;
		const int M = natom_ * nproj_;

		/*
		cublasZgemm(handle, transa, transb, N, M, K, &ONE, (cuDoubleComplex*)x, K, (cuDoubleComplex*)param_.vkb_conjugate, K, &ZERO, 
			(cuDoubleComplex*)becp, N);
		*/
		cublasZgemm(handle, transa, transb, M, N, K, &ONE, 
			(cuDoubleComplex*)param_.vkb_conjugate, K, 
			(cuDoubleComplex*)x, K, &ZERO, 
			(cuDoubleComplex*)becp, M);
	}

    print_complex(becp, 10, "becp");
    //print_complex(becp, 10, "becp");
	

	return;
    grid.x = natom_;
    block.x = nproj_;
    cal_ps_kernel<<<grid, block, 0, stream>>>(param_.deeq, becp, ps, natom_, nproj_);

    trans = CUBLAS_OP_N;
    //X K * N
    //xv K

    //print_complex(ps, 10, "ps");

    if(OpType == OperationType::FP64)
    {
        cuDoubleComplex ONE, ZERO;
        ONE.y = ZERO.y = ZERO.x = 0.0;
        ONE.x = 1.0;
        cublasZgemv(handle, trans, N_, natom_ * nproj_, &ONE, (cuDoubleComplex *)param_.vkb, N_, (cuDoubleComplex *)ps,
                    1, &ZERO, (cuDoubleComplex *)vkb_ps, 1);
    }
    else if(OpType == OperationType::FP32)
    {
        cuComplex ONE, ZERO;
        ONE.y = ZERO.y = ZERO.x = 0.0f;
        ONE.x = 1.0f;
        cublasCgemv(handle, trans, N_, natom_ * nproj_, &ONE, (cuComplex *)param_.vkb, N_, (cuComplex *)ps, 1,
                    &ZERO, (cuComplex *)vkb_ps, 1);
    }
    else
        printf("Not implemented\n");

    block.x = 128;
    grid.x = (N_ - 1) / block.x + 1;

    //print_complex(vkb_ps, 10, "vkb_ps");
    update_y_kernel<<<grid, block, 0, stream>>>(x, param_.g2_kin, fft_data, param_.gr_index, vkb_ps, hx, N_);
}

template void Hamiltonian<OperationType::FP64>::apply(const cuDoubleComplex *x, cuDoubleComplex *hx, void *buf, cublasHandle_t handle,
        cudaStream_t stream);

template void Hamiltonian<OperationType::FP32>::apply(const cuComplex *x, cuComplex *hx, void *buf, cublasHandle_t handle,
        cudaStream_t stream);

}

