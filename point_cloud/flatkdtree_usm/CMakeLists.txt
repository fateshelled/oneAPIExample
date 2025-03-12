cmake_minimum_required(VERSION 3.15)

set(CMAKE_C_COMPILER "icx" CACHE PATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "icpx" CACHE PATH "C++ compiler" FORCE)

project(example_flatkdtree)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(IntelSYCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit)

# For OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(${PROJECT_NAME}
  main.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  sycl
  Eigen3::Eigen
)

if(CUDAToolkit_FOUND)
  target_compile_options(${PROJECT_NAME} PRIVATE
    -fsycl
    -fsycl-targets=spir64,nvptx64-nvidia-cuda
  )
  target_link_options(${PROJECT_NAME} PRIVATE
    -fsycl
    -fsycl-targets=spir64,nvptx64-nvidia-cuda
  )
else()
  target_compile_options(${PROJECT_NAME} PRIVATE
    -fsycl
    -fsycl-targets=spir64
  )
  target_link_options(${PROJECT_NAME} PRIVATE
    -fsycl
    -fsycl-targets=spir64
  )
endif()
