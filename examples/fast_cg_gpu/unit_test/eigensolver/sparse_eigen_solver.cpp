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
#include <string>

#include "common.h"
#include "fft.h"

using Eigen::SelfAdjointEigenSolver;
using Eigen::GeneralizedSelfAdjointEigenSolver;
using namespace std;

class LinearTransformer {
    public:
    virtual ComplexVectorType apply(const ComplexVectorType & x) const = 0;
    virtual ComplexMatrixType applyMatrix(const ComplexMatrixType & x) const = 0;
    virtual int dim() const = 0;
};

ComplexMatrixType loadComplexMatrixData(const char * fname) {
    std::fstream in(fname);
    std::string line;
    std::getline(in, line);
    std::istringstream iss0(line);
    int N, M;
    iss0 >> N >> M;
    ComplexMatrixType ret(N, M);
    for (int i = 0; i < N; ++i) {
        std::getline(in, line);
        std::istringstream iss1(line);
        for (int j = 0; j < M; ++j) {
            iss1 >> ret(i, j);
        }
    }
    in.close();
    return ret;
}

class SparseEigenSolver {
    protected:
    ElemType _eps;
    ComplexMatrixType _X0;

    public:
    SparseEigenSolver() {}
    SparseEigenSolver(ElemType eps, bool random_init) : _eps(eps) {
        ComplexMatrixType X = loadComplexMatrixData("cg_dump_gpu/phi_in.dat");
        _X0 = X.transpose();
        if (random_init) _X0 = ComplexMatrixType::Random(_X0.rows(), _X0.cols());
        std::cout << "X init: " << _X0.rows() << "\t" << _X0.cols() << std::endl;
    }

    virtual ComplexMatrixType solve(const LinearTransformer & H, const VectorType & P, const int K) = 0;

    ElemType error(const ComplexVectorType & Hx, const ComplexVectorType & x, const ElemType lambda) {
        return (Hx - lambda * x).cwiseAbs().mean();
    }

    ElemType error(const ComplexMatrixType & HX, const ComplexMatrixType & X, const VectorType & L) {
        ElemType err = 0.0;
        for (int i = 0; i < HX.cols(); ++i) {
            err = std::max<ElemType>(err, error(HX.col(i), X.col(i), L[i]));
        }
        return err;
    }
};

void orth(ComplexMatrixType & X) {
    int K = X.cols();
    for(int i = 0; i < K; ++i) {
        ElemType ni = X.col(i).norm();
        X.col(i) /= ni;
        for (int j = i + 1; j < K ; ++j) {
            X.col(j) -= X.col(i).dot(X.col(j)) * X.col(i);
        }
    }
}

void orthSub(ComplexMatrixType & X, int start, int end) {
	/*
    for(int i = start; i < end; ++i) {
        for (int j = start; j < i; ++j) {
            X.col(i) -= X.col(j).dot(X.col(i)) * X.col(j);
			idx++;
        }
        X.col(i) /= X.col(i).norm();
    }
	*/
	static int idx = 0;
	printf("orth-%d start %d end %d\n", idx, start, end);
	for(int i = start; i < end; ++i)
	{
		X.col(i) -= X(Eigen::all, Eigen::seqN(start, i - start)) * (X(Eigen::all, Eigen::seqN(start, i - start)).adjoint() * X.col(i));
		X.col(i) /= X.col(i).norm();
	}
}

void orth(ComplexMatrixType & X, int start) {
    auto tm = NOW();
    int K = X.cols();
    for(int i = start; i < K; ++i) {
        X.col(i) -= X(Eigen::all, Eigen::seqN(0, i)) * (X(Eigen::all, Eigen::seqN(0, i)).adjoint() * X.col(i));
        /*
        for (int j = 0; j < i; ++j) {
            //X.col(i) -= X.col(j).dot(X.col(i)) * X.col(j);
            X.col(i) -= v[j] * X.col(j);
        }
        */
        X.col(i) /= X.col(i).norm();
    }
    std::cout << "orth time: " << ELAPSE(tm) << std::endl;
}

// 用施密特正交化让v和X的前K个列向量正交
void orth(const ComplexMatrixType & X, const int K, ComplexVectorType & v, bool norm = true) {
    if (K == 0) {
        if (norm) v /= v.norm();
    }
    else {
        ComplexVectorType xv = X(Eigen::all, Eigen::seqN(0, K)).adjoint() * v;
        v -= X(Eigen::all, Eigen::seqN(0, K)) * xv;
        if (norm) v /= v.norm();
    }
}

void orth(const ComplexMatrixType & X, const int K, ComplexMatrixType & Xb, int bstart = 0) {
    for (int i = bstart; i < Xb.cols(); ++i) {
        for (int k = 0; k < K; ++k) {
            //Xb.col(i) -= Xb.col(i).dot(X.col(k)) * X.col(k);
            Xb.col(i) -= X.col(k).dot(Xb.col(i)) * X.col(k);
        }
        for (int k = 0; k < i; ++k) {
            //Xb.col(i) -= Xb.col(i).dot(Xb.col(k)) * Xb.col(k);
            Xb.col(i) -= Xb.col(k).dot(Xb.col(i)) * Xb.col(k);
        }
        Xb.col(i) /= Xb.col(i).norm();
    }
}

VectorType loadVectorData(const char * fname) {
    std::fstream in(fname);
    std::string line;
    std::getline(in, line);
    std::istringstream iss0(line);
    int N;
    iss0 >> N;
    VectorType ret = VectorType::Zero(N);
    std::getline(in, line);
    std::istringstream iss1(line);
    for (int i = 0; i < N; ++i) {
        iss1 >> ret[i];
    }
    in.close();
    return ret;
}

std::vector<int> loadIndexData(const char * fname) {
    std::fstream in(fname);
    std::string line;
    std::getline(in, line);
    std::istringstream iss0(line);
    int N;
    iss0 >> N;
    std::vector<int> ret(N, 0);
    std::getline(in, line);
    std::istringstream iss1(line);
    for (int i = 0; i < N; ++i) {
        iss1 >> ret[i];
    }
    in.close();
    return ret;
}

std::vector<MatrixType> loadDeeq(const char * fname) {
    std::fstream in(fname);
    std::string line;
    std::getline(in, line);
    int nspin, natom, nproj1, nproj2;
    std::istringstream iss0(line);
    iss0 >> nspin >> natom >> nproj1 >> nproj2;
    std::vector<MatrixType> ret;
    for (int i = 0; i < natom; ++i) {
        MatrixType A = MatrixType::Zero(nproj1, nproj2);
        for(int j = 0; j < nproj1; ++j) {
            std::getline(in, line);
            std::istringstream iss(line);
            for (int k = 0; k < nproj2; ++k) {
                iss >> A(j, k);
            }
        }
        ret.push_back(A);
    }
    in.close();
    return ret;
}

void saveComplexVector(const char * fname, const ComplexVectorType & x) {
    std::ofstream out(fname);
    for(int i = 0; i < x.size(); ++i) {
        out << x[i] << std::endl;
    }
    out.close();
}

class AbacusHamiltonian : public LinearTransformer {
    private:
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
    FFT * _fft;

    public:
    AbacusHamiltonian() {
        
        _g2kin = loadVectorData("cg_dump_gpu/hamiltonian/g2kin.dat");
        _vr_eff = loadVectorData("cg_dump_gpu/hamiltonian/vr_eff.dat");
        _gr_index = loadIndexData("cg_dump_gpu/hamiltonian/GR_index.dat");
        _vkb = loadComplexMatrixData("cg_dump_gpu/hamiltonian/vkb.dat");
        _vkb_conjugate = _vkb.conjugate();
        _vkb_transpose = _vkb.transpose();

        _deeq = loadDeeq("cg_dump_gpu/hamiltonian/deeq.dat");
        _fft = new FFT(_nx, _ny, _nz);
        
#ifdef DEBUG
        std::cout << _deeq[0] << std::endl;
        std::cout << "g2kin: " << _g2kin.size() << std::endl
            << "vr_eff: " << _vr_eff.size() << std::endl
            << "_gr_index: " << _gr_index.size() << std::endl;
#endif
    }

    ~AbacusHamiltonian() {
        delete _fft;
    }

    VectorType preconditioner() {
        VectorType ret = VectorType::Zero(_g2kin.size());
        for (int i = 0; i < _g2kin.size(); ++i) {
            ElemType e = _g2kin[i];
            ret[i] = 1.0 + e + sqrt(1.0 + (e - 1) * (e - 1));
        }
        return ret;
    }

    ComplexVectorType apply(const ComplexVectorType & x) const {
        //auto start = NOW();
        int N = dim();
        ComplexVectorType y = ComplexVectorType::Zero(N);
        // g2kin
        y += x.cwiseProduct(_g2kin);

        // fft
        ComplexVectorType fft_input = ComplexVectorType::Zero(_vr_eff.size());
        for (int i = 0; i < N; ++i) {
            fft_input[_gr_index[i]] = x[i];
        }
       _fft->fft3d_all(fft_input, _vr_eff);
        for (int i = 0; i < N; ++i) {
            y[i] += fft_input[_gr_index[i]];
        }

        ComplexVectorType becp = _vkb_conjugate * x;
        int natom = _deeq.size();
        int nproj = _deeq[0].rows();
        ComplexVectorType ps = ComplexVectorType::Zero(natom * nproj);
        for (int i = 0; i < natom; ++i) {
            ps(Eigen::seqN(i*nproj, nproj)) += _deeq[i] * becp(Eigen::seqN(i*nproj, nproj));
        }
        y += _vkb_transpose * ps;
        
        return y;
    }

    ComplexMatrixType applyMatrix(const ComplexMatrixType & x) const {
        auto start = NOW();
        int N = dim();
        int K = x.cols();
        ComplexMatrixType y = ComplexMatrixType::Zero(N, K);
        ComplexVectorType fft_input = ComplexVectorType::Zero(_vr_eff.size());

        for (int ic = 0; ic < K; ++ic) {
            y.col(ic) += x.col(ic).cwiseProduct(_g2kin);

            for (int i = 0; i < N; ++i) {
                fft_input[_gr_index[i]] = x(i, ic);
            }
            _fft->fft3d_all(fft_input, _vr_eff);
            for (int i = 0; i < N; ++i) {
                y(i, ic) += fft_input[_gr_index[i]];
            }
            fft_input.fill(0);
        }
        
        ComplexMatrixType becp = _vkb_conjugate * x; //224 x K

		printf("[H apply matrix]  vkb_conjugate (%d, %d) x (%d, %d) becp (%d, %d)\n",
			_vkb_conjugate.rows(), _vkb_conjugate.cols(),
			x.rows(), x.cols(),
			becp.rows(), becp.cols());

        int natom = _deeq.size();
        int nproj = _deeq[0].rows();
        ComplexMatrixType ps = ComplexMatrixType::Zero(natom * nproj, K);
        for (int i = 0; i < natom; ++i) {
            ps(Eigen::seqN(i*nproj, nproj), Eigen::all) += _deeq[i] * becp(Eigen::seqN(i*nproj, nproj), Eigen::all);
        }
        y += _vkb_transpose * ps;

		printf("[H apply matrix]  vkb_transpose (%d, %d) ps (%d, %d) y (%d, %d)\n",
			_vkb_transpose.rows(), _vkb_transpose.cols(),
			ps.rows(), ps.cols(),
			y.rows(), y.cols());
        std::cout << "apply matrix cost: " << ELAPSE(start) << std::endl;
        return y;
    }

    int dim() const {
        return _g2kin.size();
    }
};

ElemType dot_real(const ComplexVectorType & a, const ComplexVectorType & b) {
    return a.dot(b).real();
}

class DavidsonLiu3SparseEigenSolver : public SparseEigenSolver {
    public:
    using SparseEigenSolver::SparseEigenSolver;

    ComplexMatrixType solve(const LinearTransformer & H, const VectorType & P, const int K) {
        int N = H.dim();
        ComplexMatrixType X = _X0;
        orth(X);
        VectorType L = VectorType::Zero(K);
        VectorType E = VectorType::Zero(K);
        ComplexMatrixType HX = H.applyMatrix(X);
        ComplexMatrixType XTHX = ComplexMatrixType::Zero(K, K);
        ComplexMatrixType XXTHX = ComplexMatrixType::Zero(N, K);
        ComplexMatrixType R0 = ComplexMatrixType::Zero(N, K);
        
        int M = K / 2;
        ComplexMatrixType Q = ComplexMatrixType::Random(N, M);
        ComplexMatrixType HQ = ComplexMatrixType::Zero(N, M);

        VectorType RNorm(K);
        int epoch = 0;
        for (epoch = 0; epoch < 100; ++epoch) {
            auto start = NOW();
            ElemType err = error(HX, X, L);
            if (err < _eps) {
                std::cout << "finish in " << epoch << " epoch" << std::endl;
                break;
            }
            XTHX = X.adjoint() * HX;
			cout << "X.adjoint() shape " << X.adjoint().rows() << " " << X.adjoint().cols() << endl;
			cout <<"HX " << HX.rows() << " " << HX.cols() << endl;
			cout <<"XTHX " << XTHX.rows() << " " << XTHX.cols() << endl;
            XXTHX = X * XTHX;
			printf("x.shape (%d, %d), XTHX (%d, %d) XXTHX (%d, %d)\n",
				X.rows(), X.cols(),
				XTHX.rows(), XTHX.cols(),
				XXTHX.rows(), XXTHX.cols());
			
            R0 = HX - XXTHX;
            std::vector<int> idx;
            for (int i = 0; i < K; ++i) {
                RNorm[i] = R0.col(i).cwiseAbs().mean();
                if (RNorm[i] > _eps) {
                    idx.push_back(i);
                }
            }
            int J = idx.size();
            ComplexMatrixType R = R0(Eigen::all, idx);
            for (int i = 0; i < N; ++i) {
                R.row(i).array() /= P[i];
            }
            
            ComplexMatrixType S(N, K+M+J);
            S(Eigen::all, Eigen::seqN(0, K)) = X;
            S(Eigen::all, Eigen::seqN(K, M)) = Q;
            S(Eigen::all, Eigen::seqN(K+M, J)) = R;
            if (epoch == 0) {
                auto start_orth = NOW();
                ComplexMatrixType XQTR = S(Eigen::all, Eigen::seqN(0, K)).adjoint() * S(Eigen::all, Eigen::seqN(K, M+J));
                S(Eigen::all, Eigen::seqN(K, M+J)) -= S(Eigen::all, Eigen::seqN(0, K)) * XQTR;
                orthSub(S, K, K+M+J);
                std::cout << "orth cost: " << ELAPSE(start_orth) << std::endl;

                HQ = H.applyMatrix(S(Eigen::all, Eigen::seqN(K, M)));
            }
            else {
                auto start_orth = NOW();
                ComplexMatrixType XQTR = S(Eigen::all, Eigen::seqN(0, K+M)).adjoint() * S(Eigen::all, Eigen::seqN(K+M, J));
                S(Eigen::all, Eigen::seqN(K+M, J)) -= S(Eigen::all, Eigen::seqN(0, K+M)) * XQTR;
                orthSub(S, K+M, K+M+J);
                std::cout << "orth cost: " << ELAPSE(start_orth) << std::endl;
            }
            
            ComplexMatrixType HS = ComplexMatrixType::Zero(N, K+M+J);
            HS(Eigen::all, Eigen::seqN(0, K)) = HX;
            HS(Eigen::all, Eigen::seqN(K, M)) = HQ;
            HS(Eigen::all, Eigen::seqN(K+M, J)) = H.applyMatrix(S(Eigen::all, Eigen::seqN(K+M, J)));   // HR
            
            auto start_es = NOW();
            /*
            |X'|                    | X'HX X'HQ X'HR |
            |Q'|  * |HX, HQ, HR| =  | Q'HX Q'HQ Q'HR |
            |R'|                    | R'HX R'HQ R'HR |
            */
            ComplexMatrixType STHS(K+M+J, K+M+J);
            STHS(Eigen::seqN(0, K), Eigen::seqN(0, K)) = XTHX;
            STHS(Eigen::seqN(0, K), Eigen::seqN(K, M)) = S(Eigen::all, Eigen::seqN(0, K)).adjoint() * HS(Eigen::all, Eigen::seqN(K, M));
            STHS(Eigen::seqN(K, M), Eigen::seqN(0, K)) = STHS(Eigen::seqN(0, K), Eigen::seqN(K, M)).adjoint();
            STHS(Eigen::seqN(0, K), Eigen::seqN(K+M, J)) = S(Eigen::all, Eigen::seqN(0, K)).adjoint() * HS(Eigen::all, Eigen::seqN(K+M, J));
            STHS(Eigen::seqN(K+M, J), Eigen::seqN(0, K)) = STHS(Eigen::seqN(0, K), Eigen::seqN(K+M, J)).adjoint();

            STHS(Eigen::seqN(K, M), Eigen::seqN(K, M)) = S(Eigen::all, Eigen::seqN(K, M)).adjoint() * HS(Eigen::all, Eigen::seqN(K, M));
            STHS(Eigen::seqN(K, M), Eigen::seqN(K+M, J)) = S(Eigen::all, Eigen::seqN(K, M)).adjoint() * HS(Eigen::all, Eigen::seqN(K+M, J));
            STHS(Eigen::seqN(K+M, J), Eigen::seqN(K, M)) = STHS(Eigen::seqN(K, M), Eigen::seqN(K+M, J)).adjoint();

            STHS(Eigen::seqN(K+M, J), Eigen::seqN(K+M, J)) = S(Eigen::all, Eigen::seqN(K+M, J)).adjoint() * HS(Eigen::all, Eigen::seqN(K+M, J));

            SelfAdjointEigenSolver<ComplexMatrixType> es(STHS);
            
            L = es.eigenvalues()(Eigen::seqN(0, K));
            ComplexMatrixType V = es.eigenvectors();
            
            X = S * V(Eigen::all, Eigen::seqN(0, K));
            HX = HS * V(Eigen::all, Eigen::seqN(0, K));


			printf("S (%d, %d) HS (%d, %d) HX (%d, %d) HQ (%d, %d) V (%d, %d)\n",
				S.rows(), S.cols(), 
				HS.rows(), HS.cols(),
				HX.rows(), HX.cols(),
				HQ.rows(), HQ.cols(),
				V.rows(), V.cols());

            Q = S * V(Eigen::all, Eigen::seqN(K, M));
            HQ = HS * V(Eigen::all, Eigen::seqN(K, M));
            auto cost_es = ELAPSE(start_es);
            
            std::cout << "epoch: " << epoch 
                << " J: " << J
                << " err: " << err 
                << " elapse: " << ELAPSE(start) 
                << " es cost: " << cost_es
                << std::endl << L(Eigen::seqN(0, 8)) << "\n ---- \n" << L(Eigen::lastN(8)) << std::endl;
        }
        std::cout << "eigen values: \n" << L << std::endl;
        std::cout << "epoch : " << epoch << std::endl;
        return X;
    }
};

class DavidsonLiuSparseEigenSolver : public SparseEigenSolver {
    public:
    using SparseEigenSolver::SparseEigenSolver;

    ComplexMatrixType solve(const LinearTransformer & H, const VectorType & P, const int K) {
        int N = H.dim();
        ComplexMatrixType X = _X0;
        orth(X);
        VectorType L = VectorType::Zero(K);
        VectorType E = VectorType::Zero(K);
        ComplexMatrixType HX = H.applyMatrix(X);
        ComplexMatrixType HS = ComplexMatrixType::Zero(N, K * 2);
        ComplexMatrixType XTHX = ComplexMatrixType::Zero(K, K);
        ComplexMatrixType XXTHX = ComplexMatrixType::Zero(N, K);
        ComplexMatrixType R = ComplexMatrixType::Zero(N, K);
        ComplexMatrixType S(N, K * 2);
        ComplexMatrixType STHS(K*2, K*2);
        int epoch = 0;
        for (epoch = 0; epoch < 100; ++epoch) {
            auto start = NOW();
            ElemType err = error(HX, X, L);
            if (err < _eps) {
                std::cout << "finish in " << epoch << " epoch" << std::endl;
                break;
            }
            XTHX = X.adjoint() * HX;
            XXTHX = X * XTHX;
            R = HX - XXTHX;
            for (int i = 0; i < N; ++i) {
                R.row(i).array() /= P[i];
            }
            S(Eigen::all, Eigen::seqN(0, K)) = X;
            S(Eigen::all, Eigen::seqN(K, K)) = R;
            orth(S);

            HS(Eigen::all, Eigen::seqN(0, K)) = HX;
            HS(Eigen::all, Eigen::seqN(K, K)) = H.applyMatrix(S(Eigen::all, Eigen::seqN(K, K)));   // HR
            
            STHS = S.adjoint() * HS;
            SelfAdjointEigenSolver<ComplexMatrixType> es(STHS);
            L = es.eigenvalues()(Eigen::seqN(0, K));
            ComplexMatrixType V = es.eigenvectors();

            X = S(Eigen::all, Eigen::seqN(0, K)) * V(Eigen::seqN(0, K), Eigen::seqN(0, K)) + 
                S(Eigen::all, Eigen::seqN(K, K)) * V(Eigen::seqN(K, K), Eigen::seqN(0, K));

            HX = HX * V(Eigen::seqN(0, K), Eigen::seqN(0, K)) +
                HS(Eigen::all, Eigen::seqN(K, K)) * V(Eigen::seqN(K, K), Eigen::seqN(0, K));

            std::cout << "epoch: " << epoch << " err: " << err << " elapse: " << ELAPSE(start)
                << std::endl << L(Eigen::seqN(0, 8)) << "\n ---- \n" << L(Eigen::lastN(8)) << std::endl;
        }
        std::cout << "eigen values: \n" << L << std::endl;
        std::cout << "epoch : " << epoch << std::endl;
        return X;
    }
};

class AbacusBlockCGSparseEigenSolver : public SparseEigenSolver {
    public:
    using SparseEigenSolver::SparseEigenSolver;

    void rayleigh_ritz(
        const LinearTransformer & H, 
        const ComplexMatrixType & X,
        VectorType * ritz_values,
        ComplexMatrixType * ritz_vectors
    ) {
        
        ComplexMatrixType HX = H.applyMatrix(X);
        ComplexMatrixType XTHX = X.adjoint() * HX;
        //ComplexMatrixType XTX = X.adjoint() * X;
        //std::cout << XTX(Eigen::lastN(6), Eigen::lastN(6)) << std::endl;
        //auto start = NOW();
        //GeneralizedSelfAdjointEigenSolver<ComplexMatrixType> es(XTHX, XTX);
        SelfAdjointEigenSolver<ComplexMatrixType> es(XTHX);
        
        *ritz_values = es.eigenvalues();
        *ritz_vectors = X * es.eigenvectors();
        //std::cout << "rayleigh_ritz cost: " << ELAPSE(start) << std::endl;
    }

    ComplexMatrixType solve(const LinearTransformer & H, const VectorType & P, const int K) {
        int N = H.dim();
        int max_iter = 100;
        ComplexMatrixType X = _X0;
        const int block = 4;

        for (int m = 0; m < K; m += block) {
            int B = block;
            if (K - m < B) B = K - m;
            ComplexMatrixType Xb = X(Eigen::all, Eigen::seqN(m, B));
            orth(X, m, Xb, 0);
            VectorType L0 = VectorType::Zero(B);
            VectorType E = VectorType::Zero(B);
            ComplexMatrixType S(N, B * 3);
            ComplexMatrixType R0 = ComplexMatrixType::Random(N, B);
            for (int epoch = 0; epoch < 1000; ++epoch) {
                ComplexMatrixType HX = H.applyMatrix(Xb);
                
                for (int i = 0; i < B; ++i) {
                    E[i] = (HX.col(i) -  L0[i] * Xb.col(i)).cwiseAbs().mean();
                }

                ElemType err = E.mean();
                if (err < _eps) {
                    std::cout << epoch << " new eigens:\n" << L0 << std::endl;
                    break;
                } else {
                    //std::cout << err << std::endl;
                }

                ComplexMatrixType XTHX = Xb.adjoint() * HX;
                ComplexMatrixType XXTHX = Xb * XTHX;
                ComplexMatrixType R = HX - XXTHX;
                for (int i = 0; i < N; ++i) {
                    R.row(i).array() /= P[i];
                }
                S(Eigen::all, Eigen::seqN(0, B)) = Xb;
                S(Eigen::all, Eigen::seqN(B, B)) = R;
                S(Eigen::all, Eigen::seqN(2*B, B)) = R0;
                orth(X, m, S, 0);

                //std::cout << S.adjoint() * S << std::endl;

                VectorType S_ritz_values(B * 3);
                ComplexMatrixType S_ritz_vectors(N, B * 3);
                rayleigh_ritz(H, S, &S_ritz_values, &S_ritz_vectors);
                Xb = S_ritz_vectors(Eigen::all, Eigen::seqN(0, B));
                VectorType L = S_ritz_values(Eigen::seqN(0, B));

                VectorType E = L0 - L;
                L0 = L;
                R0 = R;
            }
            X(Eigen::all, Eigen::seqN(m, B)) = Xb;
        }
        
        return X;
    }
};


class AbacusCGSparseEigenSolver : public SparseEigenSolver {
    public:
    using SparseEigenSolver::SparseEigenSolver;

    ComplexMatrixType solve(const LinearTransformer & H, const VectorType & P, const int K) {
        int N = H.dim();
        int max_iter = 100;
        ComplexMatrixType X = _X0;
        ComplexVectorType Hx(N);
        ComplexVectorType g(N);
        ComplexVectorType Px(N);
        ComplexVectorType PHx(N);
        ComplexVectorType Pg(N);
        ComplexVectorType cg = ComplexVectorType::Zero(N);
        ComplexVectorType Hcg(N);
        VectorType e = VectorType::Zero(K);
        
        for (int m = 0; m < K; ++m) {
            //auto start = NOW();
            ComplexVectorType x = X.col(m);
            orth(X, m, x);
            Hx = H.apply(x);
            e[m] = dot_real(x, Hx);

            //std::cout << "before iter: " << ELAPSE(start) << std::endl;
            ElemType gg_last = 0.0;
            ElemType cg_norm = 0.0;
            ElemType theta = 0.0;
            //start = NOW();
            VectorType P1 = P;
            int niter = 0;
            for (int iter = 0; iter < max_iter; ++iter) {
                niter++;
                ElemType err = error(Hx, x, e[m]);
                if (err < _eps) break;
                Px = x.cwiseProduct(P1.cwiseInverse());
                PHx = Hx.cwiseProduct(P1.cwiseInverse());
                g = PHx - (dot_real(x, PHx) / dot_real(x, Px)) * Px;

                //g = Hx - x.dot(Hx) * x;
                orth(X, m, g, false);

                ElemType gg_inter = dot_real(g, Pg);
                Pg = P1.cwiseProduct(g);
                ElemType gg_now = dot_real(g, Pg);

                if (iter == 0) {
                    gg_last = gg_now;
                    cg = g;
                } else {
                    const ElemType gamma = (gg_now - gg_inter) / gg_last;
                    gg_last = gg_now;
                    cg = cg * gamma + g;

                    const ElemType norma = gamma * cg_norm * sin(theta);
                    cg -= norma * x;
                }
                cg_norm = cg.norm();
                if (cg_norm < 1e-8) break;

                Hcg = H.apply(cg);
                

                const ElemType a0 = dot_real(x, Hcg) * 2.0 / cg_norm;
                const ElemType b0 = dot_real(cg, Hcg) / cg_norm / cg_norm;
                
                const ElemType e0 = e[m];
                theta = atan(a0 / (e0 - b0)) * 0.5;
                
                const ElemType new_e = (e0 - b0) * cos(2.0 * theta) + a0 * sin(2.0 * theta);
                const ElemType e1 = (e0 + b0 + new_e) * 0.5;
                const ElemType e2 = (e0 + b0 - new_e) * 0.5;
                if (e1 > e2) {
                    theta += 1.5707963;
                }
                e[m] = std::min(e1, e2);
                const ElemType cost = cos(theta);
                const ElemType sint_norm = sin(theta) / cg_norm;
                
                x = x * cost + cg * sint_norm;
                if(iter > 0) x /= x.norm();
                Hx = Hx * cost + Hcg * sint_norm;
                
            }
            //std::cout << "one iter elapse: " << ELAPSE(start) << "\t" << niter << std::endl;
            orth(X, m, x);
            X.col(m) = x;

            // 保证 e 是从小到大排序的，如果不是，则改变X的顺序
            if (m > 0 && e[m] < e[m-1]) {
                int insertIdx = 0;
                for (int i = 0; i < m; ++i) {
                    if (e[m] < e[i]) {
                        insertIdx = i;
                        break;
                    }
                }

                ElemType em = e[m];
                ComplexVectorType xm = X.col(m);
                for (int i = m; i > insertIdx; --i) {
                    X.col(i) = X.col(i-1);
                    e[i] = e[i-1];
                }
                e[insertIdx] = em;
                X.col(insertIdx) = xm;
            }
            std::cout << e[m] << "\t" << niter << std::endl;
        }
        std::cout << "eigen values:\n" << e << std::endl;
        
        return X;
    }
};

class PPCGEigenSolver : public SparseEigenSolver {
    public:
    using SparseEigenSolver::SparseEigenSolver;

    VectorType rayleigh_ritz(
        const LinearTransformer & H, 
        ComplexMatrixType & X
    ) {
        
        ComplexMatrixType HX = H.applyMatrix(X);
        ComplexMatrixType XTHX = X.adjoint() * HX;
        SelfAdjointEigenSolver<ComplexMatrixType> es(XTHX);
        X = X * es.eigenvectors();
        return es.eigenvalues();
    }

    ComplexMatrixType solve(const LinearTransformer & H, const VectorType & Pcond, const int K) {
        int N = H.dim();
        int max_iter = 100;
        ComplexMatrixType X = _X0;
        orth(X);
        ComplexMatrixType P = ComplexMatrixType::Zero(N, K);
        VectorType L = VectorType::Zero(K);
        for (int epoch = 0; epoch < 100; epoch++) {
            ComplexMatrixType HX = H.applyMatrix(X);
            ComplexMatrixType XTHX = X.adjoint() * HX;
            std::cout << L(Eigen::seqN(0, 8)) << "\n-----------\n" << L(Eigen::lastN(8)) << "\n----------\n" << std::endl;
            ComplexMatrixType W = HX - X * XTHX;
            for (int i = 0; i < N; ++i) {
                W.row(i).array() /= Pcond[i];
            }
            ComplexMatrixType XTW = X.adjoint() * W;
            W -= X * XTW;
            if (epoch == 0) {
                ComplexMatrixType S(N, 2);
                for (int j = 0; j < K; ++j) {
                    S.col(0) = X.col(j);
                    S.col(1) = W.col(j);

                    ComplexMatrixType HS(N, 2);
                    HS.col(0) = HX.col(j);
                    HS.col(1) = H.apply(S.col(1));
                    ComplexMatrixType STHS = S.adjoint() * HS;
                    ComplexMatrixType STS = S.adjoint() * S;
                    GeneralizedSelfAdjointEigenSolver<ComplexMatrixType> es(STHS, STS);
                    ComplexVectorType c = es.eigenvectors().col(0);
                    P.col(j) = c[1] * W.col(j);
                    X.col(j) = c[0] * X.col(j);
                }
            } else {
                ComplexMatrixType XTP = X.adjoint() * P;
                P -= X * XTP;
                ComplexMatrixType S(N, 3);
                for (int j = 0; j < K; ++j) {
                    S.col(0) = X.col(j);
                    S.col(1) = W.col(j);
                    S.col(2) = P.col(j);

                    ComplexMatrixType HS(N, 3);
                    HS.col(0) = HX.col(j);
                    HS(Eigen::all, Eigen::seqN(1, 2)) = H.applyMatrix(S(Eigen::all, Eigen::seqN(1, 2)));
                    ComplexMatrixType STHS = S.adjoint() * HS;
                    ComplexMatrixType STS = S.adjoint() * S;
                    GeneralizedSelfAdjointEigenSolver<ComplexMatrixType> es(STHS, STS);
                    ComplexVectorType c = es.eigenvectors().col(0);
                    P.col(j) = c[1] * W.col(j) + c[2] * P.col(j);
                    X.col(j) = c[0] * X.col(j) + P.col(j);
                }
            }
            orth(X);
            if (epoch < 5 || epoch % 5 == 0) L = rayleigh_ritz(H, X);
        }
        return X;
    }
};

int main(int argc, char ** argv) {
    std::string type = "cg";
    std::string init = "rand";
    ElemType eps = 0.001;

    if (argc > 1) type = argv[1];
    if (argc > 2) init = argv[2];
    if (argc > 3) eps = (ElemType)(atof(argv[3]));

    std::cout << type << "\t" << init << "\t" << eps << std::endl;
    
    AbacusHamiltonian H;
    SparseEigenSolver * solver;
    bool random_init = false;
    if (init == "rand") random_init = true;
    if (type == "cg") solver = new AbacusCGSparseEigenSolver(eps, random_init);
    else if (type == "dl") solver = new DavidsonLiuSparseEigenSolver(eps, random_init);
    else if (type == "dl3") solver = new DavidsonLiu3SparseEigenSolver(eps, random_init);
    else {
        std::cout << "unsupport type: " << type;
        return 0;
    }
    auto start = NOW();
    ComplexMatrixType X = solver->solve(H, H.preconditioner(), 130);
    std::cout << "elapsed: " << ELAPSE(start) << std::endl;
    return 0;
}
