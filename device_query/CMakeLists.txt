cmake_minimum_required(VERSION 3.15)

set(CMAKE_C_COMPILER "icx" CACHE PATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER "icpx" CACHE PATH "C++ compiler" FORCE)

project(device_query)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(oneMath REQUIRED)

add_executable(device_query
  main.cpp
)

target_link_libraries(device_query PRIVATE
  onemath
)
target_compile_options(device_query PRIVATE
  -fsycl
)
target_link_options(device_query PRIVATE
  -fsycl
)
