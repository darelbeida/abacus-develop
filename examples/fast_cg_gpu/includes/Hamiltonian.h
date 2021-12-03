#pragma once
#include "complex_utils.h"
#include "fft.h"
#include "common.h"
namespace faster_dft
{

template <typename ComplexType, typename ElemType>
struct HamiltonianParam
{
    const ComplexType *vkb;
    const ComplexType *vkb_conjugate;
    const ComplexType *vkb_transpose;
    const ElemType *deeq;
    const ElemType *g2_kin;
    const ElemType *vr_eff;
    const ElemType *pre_conditioner;
    const int *gr_index;
};


template<OperationType OpType>
class Hamiltonian
{
private:
    typedef Traits<OpType> Traits_;

    typedef typename Traits_::ComplexType ComplexType;
    typedef typename Traits_::ElemType ElemType;

    const int nx_, ny_, nz_, N_, n_xyz_, K_;
    const int natom_, nproj_;
    HamiltonianParam<ComplexType, ElemType> param_;
    FFT<OpType> *fft3d_;
    ComplexType *fft_data, *becp, *ps, *vkb_ps;


public:
    Hamiltonian(const int nx, const int ny, const int nz, const int N, const int K, const int natom, const int nproj1, const int nproj2):
        nx_(nx), ny_(ny), nz_(nz), n_xyz_(nx * ny * nz), N_(N), K_(K), natom_(natom), nproj_(nproj1)
    {
        fft3d_ = new FFT<OpType>(nx_, ny_, nz_, K_);

        size_t free, total;
        float scaler = 1024 * 1024 * 1024;
        cudaMemGetInfo(&free, &total);
        printf("before Hamitomain total %.2f GB, free %.2f GB\n", total / scaler, free / scaler);
        printf("Hamiltonain nx %d ny %d nz %d nxyz %d natom %d nproj %d\n", nx_, ny_, nz_, n_xyz_, natom_, nproj_);
        check_cuda_error(cudaMalloc((void **)&fft_data, sizeof(ComplexType) * n_xyz_ * K_));
        check_cuda_error(cudaMalloc((void **)&becp, sizeof(ComplexType) * natom_ * nproj_ * K_));
        check_cuda_error(cudaMalloc((void **)&ps, sizeof(ComplexType) * natom_ * nproj_ * K_));
        check_cuda_error(cudaMalloc((void **)&vkb_ps, sizeof(ComplexType) * N_ * K_));
        printf("finish Hamiltonian init\n");
        cudaMemGetInfo(&free, &total);
        printf("after Hamitomain total %.2f GB, free %.2f GB\n", total / scaler, free / scaler);
    }

    void init(HamiltonianParam<ComplexType, ElemType> param)
    {
        param_ = param;
    }
    void apply(const ComplexType *x, ComplexType *hx, void *buf, cublasHandle_t handle, cudaStream_t stream);

    ~Hamiltonian()
    {
        delete fft3d_;
        check_cuda_error(cudaFree(fft_data));
        check_cuda_error(cudaFree(becp));
        check_cuda_error(cudaFree(ps));
        check_cuda_error(cudaFree(vkb_ps));
        printf("finish Hamiltonian deconstruction\n");
    }
};

}
