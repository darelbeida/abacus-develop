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

#include "eigensolver/common.h"
#include <cuda_runtime.h>
using namespace std;
ComplexVectorType randomVector(const int N)
{
    ComplexVectorType A = ComplexVectorType::Zero(N);
	for(int i = 0; i < N; ++i)
	{
		ElemType e1 = sin(i * 0.23 + 0.35 * i) + 0.017 * ((i * i) % 100);
		ElemType e2 = sin(i * 0.71 + 0.11 * i) + 0.013 * ((i * i) % 100);
		A(i) = std::complex<ElemType>(e1, e2);
	}
    return A;
}


ComplexMatrixType randomMatrix(const int M, const int N)
{
    ComplexMatrixType A = ComplexMatrixType::Zero(M, N);
	for(int i = 0; i < M; ++i)
	{
		for(int j = 0; j < N; ++j)
		{
			ElemType e1 = sin(i * 0.23 + 0.35 * j) + 0.017 * ((i * j) % 100);
			ElemType e2 = sin(i * 0.71 + 0.11 * j) + 0.013 * ((i * j) % 100);
			A(i, j) = std::complex<ElemType>(e1, e2);
		}
	}
    return A;
}

void loadScore(const std::string &fname, double *he)
{

    cout << "loading " << fname << endl;
    std::fstream in(fname.c_str());
    std::string line;
    std::getline(in, line);
    std::istringstream iss0(line);
    int N;
    iss0 >> N;
    std::getline(in, line);
    std::istringstream iss1(line);
    for(int i = 0; i < N; ++i)
    {
        iss1 >> he[i];
    }
    in.close();
    return;
}




VectorType loadVectorData(const std::string &fname)
{

    cout << "loading " << fname << endl;
    std::fstream in(fname.c_str());
    std::string line;
    std::getline(in, line);
    std::istringstream iss0(line);
    int N;
    iss0 >> N;
    VectorType ret = VectorType::Zero(N);
    std::getline(in, line);
    std::istringstream iss1(line);
    for(int i = 0; i < N; ++i)
    {
        iss1 >> ret[i];
    }
    in.close();
    return ret;
}

std::vector<int> loadIndexData(const std::string &fname)
{
    cout << "loading " << fname << endl;
    std::fstream in(fname.c_str());
    std::string line;
    std::getline(in, line);
    std::istringstream iss0(line);
    int N;
    iss0 >> N;
    std::vector<int> ret(N, 0);
    std::getline(in, line);
    std::istringstream iss1(line);
    for(int i = 0; i < N; ++i)
    {
        iss1 >> ret[i];
    }
    in.close();
    return ret;
}

std::vector<MatrixType> loadDeeq(const std::string &fname)
{
    cout << "loading " << fname << endl;
    std::fstream in(fname.c_str());
    std::string line;
    std::getline(in, line);
    int nspin, natom, nproj1, nproj2;
    std::istringstream iss0(line);
    iss0 >> nspin >> natom >> nproj1 >> nproj2;
    std::vector<MatrixType> ret;
    for(int i = 0; i < natom; ++i)
    {
        MatrixType A = MatrixType::Zero(nproj1, nproj2);
        for(int j = 0; j < nproj1; ++j)
        {
            std::getline(in, line);
            std::istringstream iss(line);
            for(int k = 0; k < nproj2; ++k)
            {
                iss >> A(j, k);
            }
        }
        ret.push_back(A);
    }
    in.close();
    return ret;
}


ComplexMatrixType loadComplexMatrixData(const std::string &fname)
{
    cout << "loading " << fname << endl;
    std::fstream in(fname);
    std::string line;
    std::getline(in, line);
    std::istringstream iss0(line);
    int N, M;
    iss0 >> N >> M;
    ComplexMatrixType ret(N, M);
    for(int i = 0; i < N; ++i)
    {
        std::getline(in, line);
        std::istringstream iss1(line);
        for(int j = 0; j < M; ++j)
        {
            iss1 >> ret(i, j);
        }
    }
    in.close();
    return ret;
}
template <typename T>
void init_vector(T *gpu_ptr, VectorType vector)
{
    int N = vector.size();
    T *cpu_ptr = (T *)malloc(sizeof(T) * N);
    for(int i = 0; i < N; ++i)
    {
        cpu_ptr[i] = vector[i];
    }

    cudaMemcpy(gpu_ptr, cpu_ptr, sizeof(T) * N, cudaMemcpyHostToDevice);
    free(cpu_ptr);
}

template <typename T>
void init_vector(T *gpu_ptr, std::vector<MatrixType> vector)
{
    int N = vector.size() * vector[0].rows() * vector[0].cols();
    int rows = vector[0].rows();
    int cols = vector[0].cols();
    T *cpu_ptr = (T *)malloc(sizeof(T) * N);

    for(int k = 0; k < vector.size(); ++k)
    {
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < cols; ++j)
            {
                cpu_ptr[k * rows * cols + i * cols + j] = vector[k](i, j);
            }
        }
    }

    cudaMemcpy(gpu_ptr, cpu_ptr, sizeof(T) * N, cudaMemcpyHostToDevice);
    free(cpu_ptr);
}



template <typename T>
void init_vector(T *gpu_ptr, ComplexMatrixType vector)
{
    int N = vector.rows() * vector.cols();
    int rows = vector.rows();
    int cols = vector.cols();
    T *cpu_ptr = (T *)malloc(sizeof(T) * N);
    for(int i = 0; i < N; ++i)
    {
        cpu_ptr[i].x = vector(i / cols, i % cols).real();
        cpu_ptr[i].y = vector(i / cols, i % cols).imag();
    }

    cudaMemcpy(gpu_ptr, cpu_ptr, sizeof(T) * N, cudaMemcpyHostToDevice);
    free(cpu_ptr);
}

template <typename T>
void init_vector(T *gpu_ptr, ComplexVectorType vector)
{
    int N = vector.size();
    T *cpu_ptr = (T *)malloc(sizeof(T) * N);
    for(int i = 0; i < N; ++i)
    {
        cpu_ptr[i].x = vector(i).real();
        cpu_ptr[i].y = vector(i).imag();
    }

    cudaMemcpy(gpu_ptr, cpu_ptr, sizeof(T) * N, cudaMemcpyHostToDevice);
    free(cpu_ptr);
}

template <typename T>
void init_vector(T *gpu_ptr, std::vector<int> v)
{
    int N = v.size();
    T *cpu_ptr = (T *)malloc(sizeof(T) * N);
    for(int i = 0; i < N; ++i)
    {
        cpu_ptr[i] = v[i];
    }

    cudaMemcpy(gpu_ptr, cpu_ptr, sizeof(T) * N, cudaMemcpyHostToDevice);
    free(cpu_ptr);
}


template <typename T>
void check_diff(T *gpu_ptr, ComplexMatrixType v)
{
    int N = v.rows() * v.cols();
    int rows = v.rows();
    int cols = v.cols();
    T *cpu_ptr = (T *)malloc(sizeof(T) * N);

    cudaMemcpy(cpu_ptr, gpu_ptr, sizeof(T) * N, cudaMemcpyDeviceToHost);

    double diff = 0;
    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            double cur_diff = fabs(cpu_ptr[i * cols + j].x - v(i, j).real());
            if(cur_diff > diff)
                diff = cur_diff;

            cur_diff = fabs(cpu_ptr[i * cols + j].y - v(i, j).imag());
            if(cur_diff > diff)
                diff = cur_diff;
        }
    }

    cout << "diff " << diff << endl;

}


