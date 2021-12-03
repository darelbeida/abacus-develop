cmake_minimum_required(VERSION 3.8)
set(faster_dft_files
  fft.cu
  Hamiltonian.cu
  orth.cu
  eigen_solver.cu
  complex_utils.cu
)

add_definitions(-DGOOGLE_CUDA=1)
add_definitions(-DNDEBUG)

add_library(faster_dft SHARED ${faster_dft_files})
set_target_properties(faster_dft PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(faster_dft PRIVATE -lcufft -lcublas ${CMAKE_THREAD_LIBS_INIT})

