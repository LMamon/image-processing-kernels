#include <cuda_runtime_api.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <iostream>

#include "cuda_check.h"
#include "stb_image.h"
#include "stb_image_write.h"

// #define BLOCK_SIZE 16

// factored out sobel magnitude
template<int BLOCK_SIZE>
__device__ int sobelMagnitude(unsigned char tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2],
                            int ty,
                            int tx) {
                                
    // Apply Sobel X kernel to compute Gx
    int gx = (tile[ty - 1][tx + 1] + 2 * tile[ty][tx + 1] + tile[ty + 1][tx + 1]) -
        (tile[ty - 1][tx - 1] + 2 * tile[ty][tx - 1] + tile[ty + 1][tx - 1]);

    // Apply Sobel Y kernel to compute Gy
    int gy = (tile[ty - 1][tx - 1] + 2 * tile[ty - 1][tx] + tile[ty - 1][tx + 1]) -
        (tile[ty + 1][tx - 1] + 2 * tile[ty + 1][tx] + tile[ty + 1][tx + 1]);

    return min(abs(gx) + abs(gy), 255);
}

// reuse neighboring pixels across threads
// minimize redundant global memory access
template<int BLOCK_SIZE>
__global__ void smSobelKernel(unsigned char* devInput,
                            unsigned char* devOutput,
                            int w,
                            int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // clamp coordinates to valid image bounds
    int safeX = min(x, w - 1);
    int safeY = min(y, h - 1);

    // shared memory tile with 1px border for sobel neighborhood access
    __shared__ unsigned char tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // load center pixel
    tile[ty][tx] = devInput[safeY * w + safeX];
    
    // load tile border pixels
    if (threadIdx.x == 0)
        //left tile border
        tile[ty][0] = devInput[safeY * w + max(x - 1, 0)];
    
    if (threadIdx.x == BLOCK_SIZE - 1)
        //right tile border
        tile[ty][BLOCK_SIZE + 1] = devInput[safeY * w + min(x + 1, w - 1)];
    
    if (threadIdx.y == 0)
        //top tile border
        tile[0][tx] = devInput[max(y - 1, 0) * w + safeX];
    
    if (threadIdx.y == BLOCK_SIZE - 1)
        //bottom tile border
        tile[BLOCK_SIZE + 1][tx] = devInput[min(y + 1, h - 1) * w + safeX];
    
    // load tile corners
    if (threadIdx.x == 0 && threadIdx.y == 0)
        // top-left tile corner
        tile[0][0] = devInput[max(y - 1, 0) * w + max(x - 1, 0)];

    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0)
        // top-right tile corner
        tile[0][BLOCK_SIZE + 1] = devInput[max(y - 1, 0) * w + min(x + 1, w - 1)];

    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1)
        // bottom-left tile corner
        tile[BLOCK_SIZE + 1][0] = devInput[min(y + 1, h - 1) * w + max(x - 1, 0)];

    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1)
        // bottom-right tile corner
        tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = devInput[min(y + 1, h - 1) * w + min(x + 1, w - 1)];
    
    __syncthreads();
    
    
    if (x >= w || y >= h)
        return;
    int pixelIdx = y * w + x;

    // skip border pixels
    if (x < 1 || x >= w - 1 ||
        y < 1 || y >= h - 1) {
            devOutput[pixelIdx] = devInput[pixelIdx];
            return;
        }
        
    // compute edge magnitude from Gx and Gy
    int G = sobelMagnitude<BLOCK_SIZE>(tile, ty, tx);

    // store edge magnitude in output image
    devOutput[pixelIdx] = static_cast<unsigned char>(G);
}

template<int BLOCK_SIZE>
void benchmarkKernel(unsigned char* devInput, unsigned char* devOutSobel, int w, int h) {
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 threadBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((w + threadBlock.x - 1) / threadBlock.x,
            (h + threadBlock.y - 1) / threadBlock.y);

    // warmup launch
    smSobelKernel<BLOCK_SIZE><<<grid, threadBlock>>>(
        devInput,
        devOutSobel,
        w,
        h);

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaEventRecord(start));
    smSobelKernel<BLOCK_SIZE><<<grid, threadBlock>>>(
        devInput,
        devOutSobel,
        w,
        h);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::cout << BLOCK_SIZE << "x" << BLOCK_SIZE << " kernel: " << ms << " ms" << std::endl;
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void smSobelExample(const char* imagePath, bool benchmark) {
    int w, h, channels;
    unsigned char* image = stbi_load(imagePath, &w, &h, &channels, 1);

    if (!image) {
        std::cout << "Failed to load image at path: " << imagePath << std::endl;
        return;
    }
    int pixelCount = w * h;
    size_t imageBytes = pixelCount * sizeof(unsigned char);

    // host (CPU) output buffer
    unsigned char* sobelImage = nullptr;

    // device (GPU) input buffer
    unsigned char* devInput;

    // device (GPU) output buffer
    unsigned char* devOutSobel;

    //allocate CPU Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    CUDA_CHECK(cudaMallocHost(&sobelImage, imageBytes));

    // allocate GPU buffers
    CUDA_CHECK(cudaMalloc(&devInput, imageBytes));
    CUDA_CHECK(cudaMalloc(&devOutSobel, imageBytes));

    // copy data to GPU
    CUDA_CHECK(cudaMemcpy(devInput, image, imageBytes, cudaMemcpyHostToDevice));

    if (benchmark) {
        benchmarkKernel<8>(devInput, devOutSobel, w, h);
        benchmarkKernel<16>(devInput, devOutSobel, w, h);
        benchmarkKernel<32>(devInput, devOutSobel, w, h);
    } else {
        dim3 threadBlock(16, 16);
        dim3 grid((w + threadBlock.x - 1) / threadBlock.x,
                (h + threadBlock.y - 1) / threadBlock.y);

        smSobelKernel<16><<<grid, threadBlock>>>(
            devInput,
            devOutSobel,
            w,
            h);

        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // copy results to CPU
    CUDA_CHECK(cudaMemcpy(sobelImage, devOutSobel, imageBytes, cudaMemcpyDeviceToHost));

    cudaFree(devInput);
    cudaFree(devOutSobel);

    stbi_write_jpg("output/smSobel.jpg", w, h, 1, sobelImage, 100);

    cudaFreeHost(sobelImage);
    stbi_image_free(image);
    return;
}
