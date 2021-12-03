#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigenvalues>
#include <iostream>
#include <random>
#include <cmath>

#include <fstream>
#include <chrono>
#include <sstream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <sys/time.h>

#include "eigensolver/common.h"
#include "fftw.h"
#include "utils.h"
#include "../includes/Hamiltonian.h"
#include "../includes/complex_utils.h"
#include "helper.h"
using namespace faster_dft;

class AbacusHamiltonian
{
public:
    VectorType _g2kin;
    VectorType _vr_eff;
    std::vector<int> _gr_index;
    int _nx = 64;
    int _ny = 40;
    int _nz = 72;
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

        //224 = (224, 9937) x 9937
        ComplexVectorType becp = _vkb_conjugate * x;

        int natom = _deeq.size(); //28
        int nproj = _deeq[0].rows(); //8
        //deeq 28 x (8,8)

        ComplexVectorType ps = ComplexVectorType::Zero(natom * nproj);

        //ps 28x8 = deeq(28, 8x8) x becp(28, 8)

        //for 28
        for(int i = 0; i < natom; ++i)
        {
            //8
            //deeq[i] 8x8
            //becp(Eigen::seqN(i * nproj, nproj) 8x1
            ps(Eigen::seqN(i * nproj, nproj)) += _deeq[i] * becp(Eigen::seqN(i * nproj, nproj));
        }

        y += _vkb_transpose * ps;
        return y;
    }

    ComplexMatrixType applyMatrix(const ComplexMatrixType &x) const
    {
        auto start = NOW();
        int N = dim();
        int K = x.cols();
        ComplexMatrixType y = ComplexMatrixType::Zero(N, K);
        ComplexVectorType fft_input = ComplexVectorType::Zero(_vr_eff.size());

        for(int ic = 0; ic < K; ++ic)
        {
            y.col(ic) += x.col(ic).cwiseProduct(_g2kin);

            for(int i = 0; i < N; ++i)
            {
                fft_input[_gr_index[i]] = x(i, ic);
            }

            _fft->fft3d_all(fft_input, _vr_eff);

			cout << "after fft " << endl;
			cout << "K " << ic << endl;
			for(int i = 0; i < 10; ++i)	
				cout << i << " " << fft_input(i) << endl;
            for(int i = 0; i < N; ++i)
            {
                y(i, ic) += fft_input[_gr_index[i]];
            }
            fft_input.fill(0);
        }

        ComplexMatrixType becp = _vkb_conjugate * x; //224 x K

		cout << "_vkb_conjugate " << _vkb_conjugate.rows() << " " << _vkb_conjugate.cols() << endl;
		cout << "becp " << becp.rows() << " " << becp.cols() << endl;
	
		for(int i = 0; i < 10; ++i)
			cout << becp(i, 0) << endl;
	
        int natom = _deeq.size();
        int nproj = _deeq[0].rows();
        ComplexMatrixType ps = ComplexMatrixType::Zero(natom * nproj, K);
        for(int i = 0; i < natom; ++i)
        {
            ps(Eigen::seqN(i*nproj, nproj), Eigen::all) += _deeq[i] * becp(Eigen::seqN(i*nproj, nproj), Eigen::all);
        }
        y += _vkb_transpose * ps;
        std::cout << "apply matrix cost: " << ELAPSE(start) << std::endl;
        return y;
    }

    int dim() const
    {
        return _g2kin.size();
    }
};

template <OperationType OpType>
void Test()
{
    typedef Traits<OpType> Traits_;
    typedef typename Traits_::ComplexType ComplexType;
    typedef typename Traits_::ElemType ElemDataType;

    const int N = 9937;
	const int K = 5;
    AbacusHamiltonian H;
    ComplexMatrixType x = randomMatrix(N, K);
	ComplexMatrixType x_transpose = x.transpose();

    ElemDataType *d_g2kin;
    ElemDataType *d_vr_eff;
    int *d_gr_index;

    ComplexType *d_vkb;
    ComplexType *d_vkb_conjugate;
    ComplexType *d_vkb_transpose;
    ElemDataType *d_deeq;
    ComplexType *d_x;
    ComplexType *d_hx;

    cudaMalloc((void **)&d_g2kin, sizeof(ElemDataType) * H._g2kin.size());
    cudaMalloc((void **)&d_vr_eff, sizeof(ElemDataType) * H._vr_eff.size());
    cudaMalloc((void **)&d_gr_index, sizeof(int) * H._gr_index.size());
    cudaMalloc((void **)&d_vkb, sizeof(ComplexType) * H._vkb.rows() * H._vkb.cols());
    cudaMalloc((void **)&d_vkb_conjugate, sizeof(ComplexType) * H._vkb.rows() * H._vkb.cols());
    cudaMalloc((void **)&d_vkb_transpose, sizeof(ComplexType) * H._vkb.rows() * H._vkb.cols());
    cudaMalloc((void **)&d_x, sizeof(ComplexType) * x.size());
    cudaMalloc((void **)&d_hx, sizeof(ComplexType) * x.size());
    cudaMalloc((void **)&d_deeq, sizeof(ElemDataType) * H._deeq.size() * H._deeq[0].rows() * H._deeq[0].cols());

    init_vector(d_g2kin, H._g2kin);
    init_vector(d_vr_eff, H._vr_eff);
    init_vector(d_gr_index, H._gr_index);

	
    init_vector(d_x, x_transpose);

    init_vector(d_vkb_conjugate, H._vkb_conjugate);
    init_vector(d_vkb, H._vkb);
    init_vector(d_vkb_transpose, H._vkb_transpose);
    init_vector(d_deeq, H._deeq);

    HamiltonianParam<ComplexType, ElemDataType> param{d_vkb, d_vkb_conjugate, d_vkb_transpose, d_deeq, d_g2kin, d_vr_eff, NULL, d_gr_index};

    void *buf;
    int buf_size = H._nx * H._ny * H._nz;
    buf_size += H._g2kin.size();
    buf_size += 2 * H._deeq.size() * H._deeq[0].rows();
	buf_size *= K;

    printf("buf size %d\n", buf_size);
    cudaMalloc((void **)&buf, sizeof(ComplexType) * buf_size);

    Hamiltonian<OpType> *h_gpu = new Hamiltonian<OpType>(H._nx,
            H._ny, H._nz, H._g2kin.size(), K, H._deeq.size(), H._deeq[0].rows(), H._deeq[0].rows());

    h_gpu->init(param);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    h_gpu->apply(d_x, d_hx, buf, handle, stream);
    //print_complex(d_hx, 10, "GPU H.apply(x)");

    cout << "Eigen H.apply(x) " << endl;
    //ComplexVectorType eigen_y = H.apply(matrix_x.col(0));
	ComplexMatrixType eigen_matrix = H.applyMatrix(x);


	/*
	ComplexVectorType y = eigen_matrix.col(0);
    for(int i = 0; i < 10; ++i)
        cout << i << " " << eigen_y(i) << " " << y(i) << endl;
	*/

	/*

    struct timeval ss, ee;
    int ite = 10;
    for(int i = 0; i < ite; ++i)
        h_gpu->apply(d_x, d_hx, buf, handle, stream);

    cudaDeviceSynchronize();
    gettimeofday(&ss, NULL);
    for(int i = 0; i < ite; ++i)
    {
        h_gpu->apply(d_x, d_hx, buf, handle, stream);
    }
    cudaDeviceSynchronize();
    gettimeofday(&ee, NULL);
    if(OpType == OperationType::FP32)
        printf("Hamiltonian FP32 costs %.3f ms\n", diffTime(ss, ee) / ite);
    if(OpType == OperationType::FP64)
        printf("Hamiltonian FP64 costs %.3f ms\n", diffTime(ss, ee) / ite);
    double diff = 0;
    ComplexType *tmp = (ComplexType *)malloc(sizeof(ComplexType) * 9937);
    cudaMemcpy(tmp, d_hx, sizeof(ComplexType) * 9937, cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; ++i)
    {
        double cur_diff = fabs(tmp[i].x - eigen_y(i).real());
        diff = diff > cur_diff ? diff : cur_diff;
        cur_diff = fabs(tmp[i].y - eigen_y(i).imag());
        diff = diff > cur_diff ? diff : cur_diff;
    }

    cout << "diff " << diff << endl;
	*/

}

int main()
{
    //Test<OperationType::FP32>();
    Test<OperationType::FP64>();
}
