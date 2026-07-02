cmake_minimum_required(VERSION 3.18)

project(
    image-processing-kernels
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 87)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)

add_executable(image-kernels
    main.cpp

    kernels/greyscale.cu
    kernels/threshold.cu
    kernels/sobel.cu
    kernels/smSobel.cu
    kernels/medianBlur.cu
)

target_include_directories(image-kernels PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_options(image-kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(image-kernels PRIVATE
    CUDA::cudart
)
