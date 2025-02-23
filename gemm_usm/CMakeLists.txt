cmake_minimum_required(VERSION 3.15)

set(CMAKE_C_COMPILER "icx" CACHE PATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "icpx" CACHE PATH "C++ compiler" FORCE)

project(example_blas_gemm_usm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(oneMath REQUIRED)

add_executable(example_blas_gemm_usm
  gemm_usm.cpp
)

target_link_libraries(example_blas_gemm_usm PRIVATE
  onemath
)
target_compile_options(example_blas_gemm_usm PRIVATE
  -fsycl
)
target_link_options(example_blas_gemm_usm PRIVATE
  -fsycl
)

