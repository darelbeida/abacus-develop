#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>
#include <random>
#include <cmath>
#include <omp.h>

#include <fstream>
#include <chrono>
#include <sstream>
#include <vector>

#include "eigensolver/common.h"
#include "fftw.h"
#include "utils.h"
#include "../includes/eigen_solver.h"
#include "../includes/common.h"
#include "helper.h"
using namespace faster_dft;

using Eigen::SelfAdjointEigenSolver;
//using namespace std;
const int stream_N = 2;

class LinearTransformer
{
public:
    virtual ComplexVectorType apply(const ComplexVectorType &x) const = 0;
    virtual ComplexMatrixType apply(const ComplexMatrixType &x) const = 0;
    virtual int dim() const = 0;
};

class SparseEigenSolver
{
public:
    virtual ComplexMatrixType solve(const LinearTransformer &H, const LinearTransformer &P, const int K) = 0;
};

class IdentityLinearTransformer : public LinearTransformer
{
public:

    ComplexVectorType apply(const ComplexVectorType &x) const
    {
        return x;
    }

    ComplexMatrixType apply(const ComplexMatrixType &X) const
    {
        return X;
    }

    int dim() const
    {
        return -1;
    }
};

class DiagTransformer : public LinearTransformer
{
public:
    VectorType _D;
    DiagTransformer(const VectorType &D)
    {
        _D = D;
    }

    ComplexVectorType apply(const ComplexVectorType &x) const
    {
        return x.cwiseProduct(_D);
    }

    ComplexMatrixType apply(const ComplexMatrixType &x) const
    {
        return x;
    }

    int dim() const
    {
        return _D.size();
    }
};

class SimpleLinearTransformer : public LinearTransformer
{
private:
    ComplexMatrixType _A;

public:
    SimpleLinearTransformer(const ComplexMatrixType &A)
    {
        _A = A;
    }

    ComplexVectorType apply(const ComplexVectorType &x) const
    {
        return _A * x;
    }

    ComplexMatrixType apply(const ComplexMatrixType &X) const
    {
        return _A * X;
    }

    int dim() const
    {
        return _A.rows();
    }
};

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
        if(norm) v /= v.norm();
    }
}

class AbacusHamiltonian : public LinearTransformer
{
public:
    VectorType _g2kin;
    VectorType _vr_eff;
    std::vector<int> _gr_index;
    int _nx = 64;
    int _ny = 40;
    int _nz = 72;
    ComplexMatrixType _vkb;
    ComplexMatrixType _vkb_conjugate;
    ComplexMatrixType _vkb_transpose;
    std::vector<MatrixType> _deeq;
    FFT_cpu *_fft;

    AbacusHamiltonian()
    {

        _g2kin = loadVectorData("cg_dump_gpu/hamiltonian/g2kin.dat");
        _vr_eff = loadVectorData("cg_dump_gpu/hamiltonian/vr_eff.dat");
        _gr_index = loadIndexData("cg_dump_gpu/hamiltonian/GR_index.dat");
        _vkb = loadComplexMatrixData("cg_dump_gpu/hamiltonian/vkb.dat");
        _vkb_conjugate = _vkb.conjugate();
        _vkb_transpose = _vkb.transpose();

        _deeq = loadDeeq("cg_dump_gpu/hamiltonian/deeq.dat");
        _fft = new FFT_cpu(_nx, _ny, _nz);
    }

    ~AbacusHamiltonian()
    {
        delete _fft;
    }

    VectorType preconditioner()
    {
        VectorType ret = VectorType::Zero(_g2kin.size());
        for(int i = 0; i < _g2kin.size(); ++i)
        {
            ElemType e = _g2kin[i];
            ret[i] = 1.0 + e + sqrt(1.0 + (e - 1) * (e - 1));
        }
        return ret;
    }

    ComplexVectorType apply(const ComplexVectorType &x) const
    {
        int N = dim();
        ComplexVectorType y = ComplexVectorType::Zero(N);
        // g2kin
        y += x.cwiseProduct(_g2kin);
        // fft
        ComplexVectorType fft_input = ComplexVectorType::Zero(_vr_eff.size());
        for(int i = 0; i < N; ++i)
        {
            fft_input[_gr_index[i]] = x[i];
        }

        _fft->fft3d(fft_input, FFTW_BACKWARD);
        fft_input.array() *= _vr_eff.array();
        _fft->fft3d(fft_input, FFTW_FORWARD);

        for(int i = 0; i < N; ++i)
        {
            y[i] += fft_input[_gr_index[i]];
        }
        ComplexVectorType becp = _vkb_conjugate * x;

        int natom = _deeq.size();
        int nproj = _deeq[0].rows();
        ComplexVectorType ps = ComplexVectorType::Zero(natom * nproj);
        for(int i = 0; i < natom; ++i)
        {
            ps(Eigen::seqN(i*nproj, nproj)) += _deeq[i] * becp(Eigen::seqN(i*nproj, nproj));
        }

        y += _vkb_transpose * ps;
        return y;
    }

    ComplexMatrixType apply(const ComplexMatrixType &x) const
    {
        return x;
    }

    int dim() const
    {
        return _g2kin.size();
    }
};

ElemType dot_real(const ComplexVectorType &a, const ComplexVectorType &b)
{
    return a.dot(b).real();
}

class AbacusCGSparseEigenSolver : public SparseEigenSolver
{

public:
    ComplexMatrixType _X0;
    VectorType e = VectorType::Zero(130);
    AbacusCGSparseEigenSolver()
    {
        _X0 = loadComplexMatrixData("cg_dump_gpu/phi_in.dat");
    }

    ComplexMatrixType solve(const LinearTransformer &H, const LinearTransformer &P, const int K)
    {
        int N = H.dim();

        printf("H dim %d\n", N);
        int max_iter = 50;
        ComplexMatrixType X = _X0.transpose(); //ComplexMatrixType::Random(N, K);
        printf("X shape %d %d\n", _X0.rows(), _X0.cols());

        ComplexVectorType Hx(N);
        ComplexVectorType g(N);
        ComplexVectorType Pg(N);
        ComplexVectorType cg = ComplexVectorType::Zero(N);
        ComplexVectorType Hcg(N);
        ElemType eps = 1e-3;
        for(int m = 0; m < K; ++m)
        {
            ComplexVectorType x = X.col(m);
            orth(X, m, x);

            Hx = H.apply(x);
            e[m] = dot_real(x, Hx);

            ElemType gg_last = 0.0;
            ElemType cg_norm = 0.0;
            ElemType theta = 0.0;

            for(int iter = 0; iter < max_iter; ++iter)
            {
                auto start = NOW();
                g = Hx - x.dot(Hx) * x;

                orth(X, m, g, false);

                ElemType gg_inter = dot_real(g, Pg);
                Pg = P.apply(g);

                ElemType gg_now = dot_real(g, Pg);

                if(iter == 0)
                {
                    gg_last = gg_now;
                    cg = g;
                }
                else
                {
                    const ElemType gamma = (gg_now - gg_inter) / gg_last;
                    gg_last = gg_now;
                    cg = cg * gamma + g;

                    const ElemType norma = gamma * cg_norm * sin(theta);
                    cg -= norma * x;
                }

                Hcg = H.apply(cg);

                cg_norm = cg.norm();
                if(cg_norm < eps) break;

                const ElemType a0 = dot_real(x, Hcg) * 2.0 / cg_norm;
                const ElemType b0 = dot_real(cg, Hcg) / cg_norm / cg_norm;

                const ElemType e0 = e[m];
                theta = atan(a0 / (e0 - b0)) * 0.5;

                const ElemType new_e = (e0 - b0) * cos(2.0 * theta) + a0 * sin(2.0 * theta);
                const ElemType e1 = (e0 + b0 + new_e) * 0.5;
                const ElemType e2 = (e0 + b0 - new_e) * 0.5;
                if(e1 > e2)
                {
                    theta += 1.5707963;
                }
                e[m] = std::min(e1, e2);
                const ElemType cost = cos(theta);
                const ElemType sint_norm = sin(theta) /cg_norm;
                x = x * cost + cg * sint_norm;
                if(iter > 0)
                    x /= x.norm();

                if(fabs(e[m] - e0) < eps)
                {
                    break;
                }
                Hx = Hx * cost + Hcg * sint_norm;
            }
            orth(X, m, x);
            X.col(m) = x;

            // 保证 e 是从小到大排序的，如果不是，则改变X的顺序
            if(m > 0 && e[m] < e[m-1])
            {
                int insertIdx = 0;
                for(int i = 0; i < m; ++i)
                {
                    if(e[m] < e[i])
                    {
                        insertIdx = i;
                        break;
                    }
                }

                ElemType em = e[m];
                ComplexVectorType xm = X.col(m);
                for(int i = m; i > insertIdx; --i)
                {
                    X.col(i) = X.col(i-1);
                    e[i] = e[i-1];
                }
                e[insertIdx] = em;
                X.col(insertIdx) = xm;
            }
        }
        return X;
    }
};

template <OperationType OpType>
void Test()
{
    typedef Traits<OpType> Traits_;
    typedef typename Traits_::ComplexType ComplexType_;
    typedef typename Traits_::ElemType ElemType_;

    AbacusHamiltonian H;
    AbacusCGSparseEigenSolver solver;
    DiagTransformer P(H.preconditioner());

    ElemType_* d_g2kin;
    ElemType_* d_vr_eff;
    ElemType_* d_pre_conditioner;
    ElemType_* d_deeq;
    int *d_gr_index;

    ComplexType_ *d_vkb;
    ComplexType_ *d_vkb_conjugate;
    ComplexType_ *d_vkb_transpose;
    ComplexType_ *d_X0[stream_N];
    ComplexType_ *d_X0_base;
    ElemType_ *he[stream_N];
    const int N = H.dim();
    const int K = 130;

    cudaMalloc((void **)&d_g2kin, sizeof(ElemType_) * H._g2kin.size());
    cudaMalloc((void **)&d_pre_conditioner, sizeof(ElemType_) * P._D.size());
    cudaMalloc((void **)&d_vr_eff, sizeof(ElemType_) * H._vr_eff.size());
    cudaMalloc((void **)&d_gr_index, sizeof(int) * H._gr_index.size());
    cudaMalloc((void **)&d_vkb, sizeof(T) * H._vkb.rows() * H._vkb.cols());
    cudaMalloc((void **)&d_vkb_conjugate, sizeof(T) * H._vkb.rows() * H._vkb.cols());
    cudaMalloc((void **)&d_vkb_transpose, sizeof(T) * H._vkb.rows() * H._vkb.cols());
    cudaMalloc((void **)&d_deeq, sizeof(ElemType_) * H._deeq.size() * H._deeq[0].rows() * H._deeq[0].cols());
    cudaMalloc((void **)&d_X0_base, sizeof(T) * N * K);

    for(int i = 0; i < stream_N; ++i)
    {
        he[i] = (ElemType_ *)malloc(sizeof(he) * K);
        memset(he[i], 0, sizeof(ElemType_) * K);
        cudaMalloc((void **)&d_X0[i], sizeof(T) * N * K);
        init_vector(d_X0[i], solver._X0);
    }

    init_vector(d_vkb, H._vkb);
    init_vector(d_vkb_conjugate, H._vkb_conjugate);
    init_vector(d_vkb_transpose, H._vkb_transpose);
    init_vector(d_deeq, H._deeq);
    init_vector(d_g2kin, H._g2kin);
    init_vector(d_vr_eff, H._vr_eff);
    init_vector(d_gr_index, H._gr_index);
    init_vector(d_X0_base, solver._X0);
    init_vector(d_pre_conditioner, P._D);

    HamiltonianParam<ComplexType_, ElemType_> h_param{d_vkb, d_vkb_conjugate,
            d_vkb_transpose, d_deeq, d_g2kin, d_vr_eff, d_pre_conditioner, d_gr_index};

    EigenSolver<OpType> *eigen_solver_[stream_N];


    cudaStream_t stream[stream_N];
    cublasHandle_t handle[stream_N];

    #pragma omp parallel for
    for(int i = 0; i < stream_N; ++i)
    {
        cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
        cublasCreate(&handle[i]);
        cublasSetStream(handle[i], stream[i]);
        eigen_solver_[i] = new EigenSolver<OpType>(H._nx, H._ny, H._nz, H._g2kin.size(), H._deeq.size(),
                H._deeq[0].rows(), K);
        eigen_solver_[i]->init(h_param);
    }
    void *buf = NULL;

    if(OpType == OperationType::FP32)
        printf("\n\n********** Precision in FP32 ********** \n\n");
    if(OpType == OperationType::FP64)
        printf("\n\n********** Precision in FP64 ********** \n\n");

    //printf("\n\n***********************\nCPU calculation start\n\n");
    ComplexMatrixType X = solver.solve(H, P, 130);
    //printf("\n\n***********************\nGPU calculation start\n\n");

    for(int i = 0; i < stream_N; ++i)
        eigen_solver_[i]->solve(d_X0[i], he[i], buf, K, handle[i], stream[i]);

    ElemType_ diff = 0.0;
    cout << "final e " << endl;

    #pragma omp parallel for
    for(int idx = 0; idx < stream_N; idx++)
    {
        cudaStreamSynchronize(stream[idx]);
        for(int i = 0; i < K; ++i)
        {
            printf("i %d cpu %lf gpu %lf diff %lf\n", i, solver.e[i], he[idx][i], fabs(solver.e[i] - he[idx][i]));
            if(fabs(solver.e[i] - he[idx][i]) > diff)
                diff = fabs(solver.e[i] - he[idx][i]);
        }
        printf("Stream %d max diff %lf\n", idx, diff);
    }

    struct timeval ss, ee;
    int ite = 5;

    //warm up
    #pragma omp parallel for
    for(int idx = 0; idx < stream_N; ++idx)
    {
        for(int i = 0; i < ite; ++i)
        {
            cudaMemcpyAsync(d_X0[idx], d_X0_base, sizeof(T) * N * K, cudaMemcpyDeviceToDevice, stream[idx]);
            memset(he[idx], 0, sizeof(ElemType_) * K);
            eigen_solver_[idx]->solve(d_X0[idx], he[idx], buf, K, handle[idx], stream[idx]);
        }
    }

    cudaDeviceSynchronize();
    gettimeofday(&ss, NULL);
    #pragma omp parallel for
    for(int idx = 0; idx < stream_N; ++idx)
    {
        for(int i = 0; i < ite; ++i)
        {
            cudaMemcpyAsync(d_X0[idx], d_X0_base, sizeof(T) * N * K, cudaMemcpyDeviceToDevice, stream[idx]);
            memset(he[idx], 0, sizeof(ElemType_) * K);
            eigen_solver_[idx]->solve(d_X0[idx], he[idx], buf, K, handle[idx], stream[idx]);
        }
    }

    cudaDeviceSynchronize();
    gettimeofday(&ee, NULL);

    if(OpType == OperationType::FP32)
        printf("EigenSolver FP32 costs %.3f ms\n", diffTime(ss, ee) / ite);
    if(OpType == OperationType::FP64)
        printf("EigenSolver FP64 costs %.3f ms\n", diffTime(ss, ee) / ite);

    for(int i = 0; i < stream_N; ++i)
        delete eigen_solver_[i];
}

int main()
{
    Test<OperationType::FP32>();
    printf("\n\n");
    Test<OperationType::FP64>();
}
