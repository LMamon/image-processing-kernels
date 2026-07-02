# image-processing-kernels

Collection of CUDA image processing kernels implemented in C++ for NVIDIA GPUs (CC 8.7).

Includes grayscale conversion, binary thresholding, naive Sobel edge detection, shared-memory tiled Sobel, and variable-size median blur. Each kernel includes a standalone example demonstrating image processing and optional runtime benchmarking.

Designed for modern CUDA toolchains and NVIDIA Jetson platforms, but compatible with any CUDA-capable GPU supporting CUDA C++.

All kernel implementations are contained in `kernels/`, public interfaces are provided in `include/`, and `main.cpp` serves as a sample command-line driver for running each kernel and benchmarking individual implementations.