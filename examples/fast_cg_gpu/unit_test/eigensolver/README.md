Eigen value/vector program for large scale sparse matrix.

# Build deps

*  FFTW3

Option 1: 

Run command `apt-get install libfftw3-dev` in Debian.

Option 2:

Download FFTW3 from `http://www.fftw.org/fftw-3.3.10.tar.gz`.

Compile and Install FFTW3 by `./configure --prefix=<somedir> --enable-single && make -j4 && make install`.

* Eigen3

Download Eigen3 package from `https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz`.

Uncompress and soft link to this project root dir: `cd eigensolver && ln -s <somedir>/eigen-3.4.0 eigen3`

# Build Eigensolver

Run `make all` directly. After compiling, you will get `sparse_eigen_solver` and `sparse_eigen_solver_single` execuable objects in the compile dir.

# Run and check values

Run `sparse_eigen_solver` for double precision and `sparse_eigen_solver_single` for single precision. Running results (eigen values) will print onto screen. Both of them should return almost the same eigen values.

# More details

File `sparse_eigen_solver.cpp` implements the Conjugate gradient method on Hamiltonian matrix. While computing Hamiltonian matrix, it calls `fft3d()` in `fft.h` each round.