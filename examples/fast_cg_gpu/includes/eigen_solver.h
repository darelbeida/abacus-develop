#pragma once
#include "Hamiltonian.h"
#include "complex_utils.h"
namespace faster_dft
{

template <OperationType OpType>
class EigenSolver
{
private:
    typedef Traits<OpType> Traits_;

    typedef typename Traits_::ComplexType ComplexType;
    typedef typename Traits_::ElemType ElemType;

    const int nx_, ny_, nz_, N_;
    const int natom_, nproj_;
    const double eps_;
	const int random_init_;
    const int max_ite_ = 50;
    const int max_double_gpu_tmp = 10;
    const int max_K_;
    Hamiltonian<OpType> *hamiltion_layer_;
    const ElemType *pre_conditioner_;

    ComplexType *Hx;
    ComplexType *x_col;
    ComplexType *g;
    ComplexType *Pg;
    ComplexType *cg;
    ComplexType *Hcg;
    ComplexType *X_buf;
    //ComplexType *x_Hx_dot;
    ElemType *double_gpu_tmp;
    ElemType *gg_buf;
    void *orth_buf;


public:
    EigenSolver(const int nx, const int ny, const int nz, const int N, const int natom, const int nproj, const int max_K, 
		const double eps, const int random_init):
        nx_(nx), ny_(ny), nz_(nz), N_(N), natom_(natom), nproj_(nproj), max_K_(max_K), eps_(eps), random_init_(random_init)
    {
		printf("EigenSolver construction nx %d ny %d nz %d N %d max_K %d natom %d nproj %d eps %lf random_init %d\n", 
				nx_, ny_, nz_, N_, max_K_,
				natom_, nproj_, eps_, random_init_);

        hamiltion_layer_ = new Hamiltonian<OpType>(nx, ny, nz, N, 1, natom, nproj, nproj);

        float scaler = 1024 * 1024 * 1024;
        size_t total, free;
        cudaMemGetInfo(&free, &total);
        printf("before Eigen total %.2f GB, free %.2f GB\n", total / scaler, free / scaler);
        check_cuda_error(cudaMalloc((void **)&Hx, sizeof(ComplexType) * N_));
        check_cuda_error(cudaMalloc((void **)&x_col, sizeof(ComplexType) * N_));
        check_cuda_error(cudaMalloc((void **)&g, sizeof(ComplexType) * N_));
        check_cuda_error(cudaMalloc((void **)&Pg, sizeof(ComplexType) * N_));
        check_cuda_error(cudaMalloc((void **)&cg, sizeof(ComplexType) * N_));
        check_cuda_error(cudaMalloc((void **)&Hcg, sizeof(ComplexType) * N_));
        check_cuda_error(cudaMalloc((void **)&X_buf, sizeof(ComplexType) * N_ * max_K));
        check_cuda_error(cudaMalloc((void **)&double_gpu_tmp, sizeof(ElemType) * max_double_gpu_tmp));
        check_cuda_error(cudaMalloc((void **)&gg_buf, sizeof(ElemType) * 3));
        check_cuda_error(cudaMalloc((void **)&orth_buf, sizeof(ComplexType) * (max_K + 1)));
        cudaMemGetInfo(&free, &total);
        printf("after Eigen total %.2f GB, free %.2f GB\n", total / scaler, free / scaler);
    }
    void init(HamiltonianParam<ComplexType, ElemType> param)
    {
        pre_conditioner_ = param.pre_conditioner;
        hamiltion_layer_->init(param);
    }
    void reset_buf(cudaStream_t stream)
    {
        check_cuda_error(cudaMemsetAsync(Hx, 0, sizeof(ComplexType) * N_, stream));
        check_cuda_error(cudaMemsetAsync(x_col, 0, sizeof(ComplexType) * N_, stream));
        check_cuda_error(cudaMemsetAsync(g, 0, sizeof(ComplexType) * N_, stream));
        check_cuda_error(cudaMemsetAsync(Pg, 0, sizeof(ComplexType) * N_, stream));
        check_cuda_error(cudaMemsetAsync(cg, 0, sizeof(ComplexType) * N_, stream));
        check_cuda_error(cudaMemsetAsync(Hcg, 0, sizeof(ComplexType) * N_, stream));
        check_cuda_error(cudaMemsetAsync(X_buf, 0, sizeof(ComplexType) * N_ * max_K_, stream));
        check_cuda_error(cudaMemsetAsync(double_gpu_tmp, 0, sizeof(ElemType) * max_double_gpu_tmp, stream));
        check_cuda_error(cudaMemsetAsync(gg_buf, 0, sizeof(ElemType) * 3, stream));
        check_cuda_error(cudaMemsetAsync(orth_buf, 0, sizeof(ComplexType) * (max_K_ + 1), stream));

    }
    void solve(ComplexType *X, ElemType *he, void *buf, const int K, cublasHandle_t handle, cudaStream_t stream);

    ~EigenSolver()
    {
        printf("Inside EigenSolver Deconstruction\n");
        check_cuda_error(cudaFree(Hx));
        check_cuda_error(cudaFree(x_col));
        check_cuda_error(cudaFree(g));
        check_cuda_error(cudaFree(Pg));
        check_cuda_error(cudaFree(cg));
        check_cuda_error(cudaFree(Hcg));
        check_cuda_error(cudaFree(X_buf));
        check_cuda_error(cudaFree(double_gpu_tmp));
        check_cuda_error(cudaFree(gg_buf));
        check_cuda_error(cudaFree(orth_buf));
        delete hamiltion_layer_;
    }
};

}
