#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>
#include <random>
#include <cmath>

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
using namespace std;

struct Config
{
    int nx, ny, nz;
    int K;
    double eps;
    int random;
    string path;
} config;

ElemType error(const ComplexVectorType &Hx, const ComplexVectorType &x, const ElemType lambda)
{
    ElemType error_val = (Hx - lambda * x).cwiseAbs().mean();

    return error_val;
}

ElemType error(const ComplexMatrixType &HX, const ComplexMatrixType &X, const VectorType &L)
{
    ElemType err = 0.0;
    for(int i = 0; i < HX.cols(); ++i)
    {
        err = std::max<ElemType>(err, error(HX.col(i), X.col(i), L[i]));
    }
    return err;
}

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
    virtual ComplexMatrixType solve(const LinearTransformer &H, const VectorType &P, const int K) = 0;
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

    int _nx = config.nx;
    int _ny = config.ny;
    int _nz = config.nz;
    /*
    int _nx = 144;
    int _ny = 144;
    int _nz = 144;
    */
    ComplexMatrixType _vkb;
    ComplexMatrixType _vkb_conjugate;
    ComplexMatrixType _vkb_transpose;
    std::vector<MatrixType> _deeq;
    FFT_cpu *_fft;

    AbacusHamiltonian()
    {

        /*
        _g2kin = loadVectorData("cg_dump_gpu/hamiltonian/g2kin.dat");
        _vr_eff = loadVectorData("cg_dump_gpu/hamiltonian/vr_eff.dat");
        _gr_index = loadIndexData("cg_dump_gpu/hamiltonian/GR_index.dat");
        _vkb = loadComplexMatrixData("cg_dump_gpu/hamiltonian/vkb.dat");
        */
        _g2kin = loadVectorData(config.path + "/hamiltonian/g2kin.dat");
        _vr_eff = loadVectorData(config.path + "/hamiltonian/vr_eff.dat");
        _gr_index = loadIndexData(config.path + "/hamiltonian/GR_index.dat");
        _vkb = loadComplexMatrixData(config.path + "/hamiltonian/vkb.dat");
        _vkb_conjugate = _vkb.conjugate();
        _vkb_transpose = _vkb.transpose();

        //_deeq = loadDeeq("cg_dump_gpu/hamiltonian/deeq.dat");
        _deeq = loadDeeq(config.path + "/hamiltonian/deeq.dat");
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

        /*
        cout << "x " << endl;
        for(int i = 0; i < 10; ++i)
            cout << i << " " << x(i) << endl;

        cout << "becp " << endl;
        for(int i = 0; i < 10; ++i)
            cout << i << " " << becp(i) << endl;
        */

        int natom = _deeq.size();
        int nproj = _deeq[0].rows();
        ComplexVectorType ps = ComplexVectorType::Zero(natom * nproj);
        for(int i = 0; i < natom; ++i)
        {
            ps(Eigen::seqN(i*nproj, nproj)) += _deeq[i] * becp(Eigen::seqN(i*nproj, nproj));
        }

        /*
        cout << "ps " << endl;
        for(int i = 0; i < 10; ++i)
            cout << i << " " << ps(i) << endl;
        cout << "vkb_ps " << endl;
        for(int i = 0; i < 10; ++i)
            cout << i << " " << (_vkb_transpose * ps)(i) << endl;
        */
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
    VectorType e = VectorType::Zero(config.K);
    AbacusCGSparseEigenSolver()
    {
        //_X0 = loadComplexMatrixData("cg_dump_gpu/phi_in.dat");
        _X0 = loadComplexMatrixData(config.path + "/phi_in.dat");
        if(config.random == 1)
            _X0 = ComplexMatrixType::Random(_X0.rows(), _X0.cols());
    }

    ComplexMatrixType solve(const LinearTransformer &H, const VectorType &P, const int K)
    {
        int N = H.dim();
        int max_iter = 50;
        ComplexMatrixType X = _X0.transpose(); //ComplexMatrixType::Random(N, K);

        ComplexVectorType Hx = ComplexVectorType::Zero(N);
        ComplexVectorType g = ComplexVectorType::Zero(N);
        ComplexVectorType Pg = ComplexVectorType::Zero(N);
        ComplexVectorType Px = ComplexVectorType::Zero(N);
        ComplexVectorType PHx = ComplexVectorType::Zero(N);
        ComplexVectorType cg = ComplexVectorType::Zero(N);
        ComplexVectorType Hcg = ComplexVectorType::Zero(N);
        ElemType eps = 1e-3;
        int iter_cnt = 0;
        for(int m = 0; m < K; ++m)
        {
            cout << "m " << m << endl;
            ComplexVectorType x = X.col(m);

            /*
            cout << "x col" << endl;
            for(int i = 0; i < 10; ++i)
                cout << i << " " << x(i) << endl;
            */
            orth(X, m, x);

            /*
            cout << "after orth" << endl;
            for(int i = 0; i < 10; ++i)
                cout << i << " " << x(i) << endl;
            */

            Hx = H.apply(x);
            e[m] = dot_real(x, Hx);
            /*
            cout << "Hx" << endl;
            for(int i = 0; i < 10; ++i)
                cout << i << " " << Hx(i) << endl;
            cout << "em " << e[m] << endl;
            */

            ElemType gg_last = 0.0;
            ElemType cg_norm = 0.0;
            ElemType theta = 0.0;

            for(int iter = 0; iter < max_iter; ++iter)
            {
                cout << "iter " << iter << endl;
                if(config.random == 1)
                {
                    ElemType err = error(Hx, x, e[m]);
                    if(err < config.eps)
                        break;
                }
                iter_cnt++;
                //g = Hx - x.dot(Hx) * x;
                Px = x.cwiseProduct(P.cwiseInverse());
                PHx = Hx.cwiseProduct(P.cwiseInverse());
                g = PHx - (dot_real(x, PHx) / dot_real(x, Px)) * Px;

                orth(X, m, g, false);

                /*
                cout << "after orth g " << endl;
                for(int i = 0; i < 10; ++i)
                    cout << i << " " << g(i) << endl;
                */

                ElemType gg_inter = dot_real(g, Pg);
                //Pg = P.apply(g);
                Pg = P.cwiseProduct(g);

                /*
                cout << "Pg " << endl;
                for(int i = 0; i < 10; ++i)
                    cout << i << " " << Pg(i) << endl;
                */

                ElemType gg_now = dot_real(g, Pg);
                //cout << "gg_now " << gg_now << endl;

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
                //printf("m-%d ite %d cg_norm %lf\n", m, iter, cg_norm);
                if(config.random == 0)
                {
                    if(cg_norm < config.eps) break;
                }

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
        cout << "CPU eigen_sovler iter_cnt " << iter_cnt << endl;
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

    ElemType_* d_g2kin;
    ElemType_* d_vr_eff;
    ElemType_* d_pre_conditioner;
    ElemType_* d_deeq;
    int *d_gr_index;

    ComplexType_ *d_vkb;
    ComplexType_ *d_vkb_conjugate;
    ComplexType_ *d_vkb_transpose;
    ComplexType_ *d_X0;
    ComplexType_ *d_X0_base;
    ComplexType_ *d_eigen_values;
    const int N = H.dim();
    const int K = config.K;

    float scaler = 1024 * 1024 * 1024;
    size_t total, free;
    cudaMemGetInfo(&free, &total);
    printf("before cudaMalloc total %.2f GB, free %.2f GB\n", total / scaler, free / scaler);

    cudaMalloc((void **)&d_g2kin, sizeof(ElemType_) * H._g2kin.size());
    cudaMalloc((void **)&d_pre_conditioner, sizeof(ElemType_) * H.preconditioner().size());
    cudaMalloc((void **)&d_vr_eff, sizeof(ElemType_) * H._vr_eff.size());
    cudaMalloc((void **)&d_deeq, sizeof(ElemType_) * H._deeq.size() * H._deeq[0].rows() * H._deeq[0].cols());
    cudaMalloc((void **)&d_gr_index, sizeof(int) * H._gr_index.size());


    cudaMalloc((void **)&d_vkb, sizeof(ComplexType_) * H._vkb.rows() * H._vkb.cols());
    cudaMalloc((void **)&d_vkb_conjugate, sizeof(ComplexType_) * H._vkb.rows() * H._vkb.cols());
    cudaMalloc((void **)&d_vkb_transpose, sizeof(ComplexType_) * H._vkb.rows() * H._vkb.cols());
    cudaMalloc((void **)&d_X0, sizeof(ComplexType_) * N * K);
    cudaMalloc((void **)&d_X0_base, sizeof(ComplexType_) * N * K);
    cudaMalloc((void **)&d_eigen_values, sizeof(ComplexType_) * K);

    init_vector(d_vkb, H._vkb);
    init_vector(d_vkb_conjugate, H._vkb_conjugate);
    init_vector(d_vkb_transpose, H._vkb_transpose);
    init_vector(d_deeq, H._deeq);
    init_vector(d_g2kin, H._g2kin);
    init_vector(d_vr_eff, H._vr_eff);
    init_vector(d_gr_index, H._gr_index);
    init_vector(d_X0, solver._X0);
    init_vector(d_X0_base, solver._X0);
    init_vector(d_pre_conditioner, H.preconditioner());

    HamiltonianParam<ComplexType_, ElemType_> h_param{d_vkb, d_vkb_conjugate,
            d_vkb_transpose, d_deeq, d_g2kin, d_vr_eff, d_pre_conditioner, d_gr_index};


    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    void *buf = NULL;

    if(OpType == OperationType::FP32)
        printf("\n\n********** Precision in FP32 ********** \n\n");
    if(OpType == OperationType::FP64)
        printf("\n\n********** Precision in FP64 ********** \n\n");

    //printf("\n\n***********************\nCPU calculation start\n\n");
    ComplexMatrixType X = solver.solve(H, H.preconditioner(), K);
    //printf("\n\n***********************\nGPU calculation start\n\n");

    //for(int i = 0; i < K; ++i)
    //  cout << i << " " << solver.e[i] << endl;

    ElemType_ *he = (ElemType_ *)malloc(sizeof(he) * K);
    memset(he, 0, sizeof(ElemType_) * K);
    EigenSolver<OpType> *eigen_solver_ = new EigenSolver<OpType>(H._nx, H._ny, H._nz, H._g2kin.size(), H._deeq.size(),
            H._deeq[0].rows(), K, config.eps, config.random);
    eigen_solver_->init(h_param);

    struct timeval ss, ee;
    cudaDeviceSynchronize();
    gettimeofday(&ss, NULL);
    cudaMemcpyAsync(d_X0, d_X0_base, sizeof(ComplexType_) *  N * K, cudaMemcpyDeviceToDevice, stream);
    eigen_solver_->solve(d_X0, he, buf, K, handle, stream);
    cudaDeviceSynchronize();
    gettimeofday(&ee, NULL);

    printf("GPU costs %.2f ms\n", diffTime(ss, ee));

    ElemType_ diff = 0.0;
    cout << "final e " << endl;

    double* cpu_tmp = (double*)malloc(sizeof(double) * K);
    loadScore("cg_dump_gpu/e_out.dat", cpu_tmp);
    for(int i = 0; i < K; ++i)
    {

        //printf("i %d cpu %lf gpu %lf diff %lf\n", i, solver.e[i], he[i], fabs(solver.e[i] - he[i]));
        //if(fabs(solver.e[i] - he[i]) > diff)
        //    diff = fabs(solver.e[i] - he[i]);
        printf("i %d cpu %lf gpu %lf diff %lf\n", i, cpu_tmp[i], he[i], fabs(cpu_tmp[i] - he[i]));
        if(fabs(cpu_tmp[i] - he[i]) > diff)
            diff = fabs(cpu_tmp[i] - he[i]);

    }
    printf("max diff %lf\n", diff);

    return;
    cudaDeviceSynchronize();
    gettimeofday(&ss, NULL);
    for(int i = 0; i < 10; ++i)
    {
        cudaMemcpyAsync(d_X0, d_X0_base, sizeof(ComplexType_) *  N * K, cudaMemcpyDeviceToDevice, stream);
        eigen_solver_->solve(d_X0, he, buf, K, handle, stream);
    }
    cudaDeviceSynchronize();
    gettimeofday(&ee, NULL);
    printf("GPU costs %.2f ms\n", diffTime(ss, ee) / 10);

    delete eigen_solver_;
}

int main(int argc, char *argv[])
{
    FILE *fd = fopen("config.in", "r");
    if(fd == NULL)
    {
        printf("File %s not found\n", "config.in");
        return 0 ;
    }
    int nx, ny, nz, K;
    double eps;
    int random;
    char path[20];

    fscanf(fd, "%d %*s", &nx);
    fscanf(fd, "%d %*s", &ny);
    fscanf(fd, "%d %*s", &nz);
    fscanf(fd, "%d %*s", &K);
    fscanf(fd, "%d %*s", &random);
    fscanf(fd, "%lf %*s", &eps);
    fscanf(fd, "%s", &path);

    fclose(fd);
    config.nx = nx;
    config.ny = ny;
    config.nz = nz;
    config.K = K;
    config.random = random;
    config.eps = eps;
    config.path = string(path);

    printf("nx %d ny %d nz %d K %d random %d eps %lf path %s\n", config.nx, config.ny, config.nz, config.K, config.random, config.eps, config.path);
    cout << config.path << endl;

    //Test<OperationType::FP32>();
    //printf("\n\n");
    Test<OperationType::FP64>();
}
