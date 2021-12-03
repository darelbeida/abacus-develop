#ifndef COMMON_H
#define COMMON_H

#include <complex>
#include <eigen3/Eigen/Dense>

#define NOW() std::chrono::system_clock::now()
#define ELAPSE(x) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - x).count()

#ifdef SINGLE
typedef Eigen::MatrixXf MatrixType;
typedef Eigen::VectorXf VectorType;
typedef Eigen::VectorXcf ComplexVectorType;
typedef Eigen::MatrixXcf ComplexMatrixType;
typedef float ElemType;
typedef std::complex<float> ComplexElemType;

#else
typedef Eigen::MatrixXd MatrixType;
typedef Eigen::VectorXd VectorType;
typedef Eigen::VectorXcd ComplexVectorType;
typedef Eigen::MatrixXcd ComplexMatrixType;
typedef double ElemType;
typedef std::complex<double> ComplexElemType;
#endif

#endif
