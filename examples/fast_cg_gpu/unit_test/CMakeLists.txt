cmake_minimum_required(VERSION 3.8)

#set(fft_gpu_files
#  fft_gpu_test.cc
#)
#add_executable(fft_test ${fft_gpu_files})
#target_link_libraries(fft_test PUBLIC -lcublas -lcudart fft ${CMAKE_THREAD_LIBS_INIT})

#set(cufft_files
#  cufft_test.cc
#)
#add_executable(cufft_test ${cufft_files})
#target_link_libraries(cufft_test PUBLIC -lcublas -lcudart -lcufft fft ${CMAKE_THREAD_LIBS_INIT})

set(gemm_bench_files
  gemm_bench_test.cc
)
add_executable(gemm_bench_test ${gemm_bench_files})
target_link_libraries(gemm_bench_test PUBLIC -lcublas -lcudart faster_dft ${CMAKE_THREAD_LIBS_INIT})

set(orth_files
  orth_gpu_test.cc
)
add_executable(orth_test ${orth_files})
target_link_libraries(orth_test PUBLIC -lcublas -lcudart faster_dft ${CMAKE_THREAD_LIBS_INIT})

set(Hamiltonian_files
  Hamiltonian_test.cc
)
add_executable(Hamiltonian_test ${Hamiltonian_files})
target_link_libraries(Hamiltonian_test PUBLIC -lcublas -lcudart -lfftw3 -lcufft -std=c++11 -O3 faster_dft ${CMAKE_THREAD_LIBS_INIT})

set(eigen_solver_files
  eigen_solver_test.cc
)
add_executable(eigen_solver_test ${eigen_solver_files})
target_link_libraries(eigen_solver_test PUBLIC -lcublas -lcudart -lfftw3 -lcufft -std=c++11 -O3 faster_dft ${CMAKE_THREAD_LIBS_INIT})
