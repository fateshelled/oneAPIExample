cmake_minimum_required(VERSION 3.15)

set(CMAKE_C_COMPILER "icx" CACHE PATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "icpx" CACHE PATH "C++ compiler" FORCE)

project(example_sort)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(IntelSYCL REQUIRED)

add_executable(${PROJECT_NAME}
  main.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  sycl
)
target_compile_options(${PROJECT_NAME} PRIVATE
  -fsycl
  -fsycl-targets=spir64,nvptx64-nvidia-cuda
)
target_link_options(${PROJECT_NAME} PRIVATE
  -fsycl
  -fsycl-targets=spir64,nvptx64-nvidia-cuda
)

