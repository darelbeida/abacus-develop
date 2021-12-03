all: sparse_eigen_solver debug_sparse_eigen_solver sparse_eigen_solver_single

sparse_eigen_solver: sparse_eigen_solver.o
	g++ -std=c++11 -o sparse_eigen_solver -O3 sparse_eigen_solver.o -lfftw3 -lm -march=native

sparse_eigen_solver.o: sparse_eigen_solver.cpp
	g++ -std=c++11 -O3 -c sparse_eigen_solver.cpp -I./ -march=native

sparse_eigen_solver_single: sparse_eigen_solver_single.o
	g++ -std=c++11 -o sparse_eigen_solver_single -O3 sparse_eigen_solver_single.o -lfftw3f -lm -march=native

sparse_eigen_solver_single.o: sparse_eigen_solver.cpp
	g++ -std=c++11 -DSINGLE -O3 -c sparse_eigen_solver.cpp -I./ -march=native -o sparse_eigen_solver_single.o

debug_sparse_eigen_solver: debug_sparse_eigen_solver.o
	g++ -std=c++11 -o debug_sparse_eigen_solver -O3 debug_sparse_eigen_solver.o -lfftw3 -lm

debug_sparse_eigen_solver.o: sparse_eigen_solver.cpp
	g++ -std=c++11 -DDEBUG -I./ -c sparse_eigen_solver.cpp -o debug_sparse_eigen_solver.o

clean:
	rm *.o sparse_eigen_solver debug_sparse_eigen_solver