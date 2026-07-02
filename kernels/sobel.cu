#include <cuda_runtime_api.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <iostream>

#define CUDA_EXPRESSION_CHECKER
#include "cuda_check.h"

#include "stb_image.h"
#include "stb_image_write.h"


__global__ void sobelKernel(unsigned char* devInput,
                            unsigned char* devOutput,
                            int w,
                            int h) {
    // calculate pixel index handled by this thread
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelCount = w * h;
    if (pixelIdx >= pixelCount)
        return;

    // Convert pixel index to (x,y)
    int x = pixelIdx % w;
    int y = pixelIdx / w;
    constexpr int r = 1;
    
    if (x < r || x >= w - r ||
        y < r || y >= h - r) {
            devOutput[pixelIdx] = devInput[pixelIdx];
            return;
        }
    
    // read 3x3 neighborhood around current pixel
    unsigned char window[3][3];

    for (int dy = -r; dy <= r; dy++) {
        for (int dx = -r; dx <= r; dx++){
            int row = dy + r;
            int col = dx + r;

            int neighborX = x + dx;
            int neighborY = y + dy;

            int neighborIdx = neighborY * w + neighborX;

            window[row][col] = devInput[neighborIdx];
        }
    }

    // Apply Sobel X kernel to compute Gx
    int gx = (window[0][2] + 2 * window[1][2] + window[2][2]) -
                        (window[0][0] + 2 * window[1][0] + window[2][0]);

    // Apply Sobel Y kernel to compute Gy
    int gy = (window[0][0] + 2 * window[0][1] + window[0][2]) -
                        (window[2][0] + 2 * window[2][1] + window[2][2]);
        
    int G = abs(gx) + abs(gy);
    if (G > 255)
        G = 255;

    devOutput[pixelIdx] = static_cast<unsigned char>(G);
}

void benchmarkSobelKernel(unsigned char* devInput, unsigned char* devOutput, int w, int h) {
    constexpr int threads = 256;
    const int pixelCount = w * h;
    const int blocks = (pixelCount + threads - 1) / threads;

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup launch
    sobelKernel<<<blocks, threads>>>(devInput,
                                    devOutput,
                                    w,
                                    h);

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    sobelKernel<<<blocks, threads>>>(devInput,
                                    devOutput,
                                    w,
                                    h);

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "\nKernel: Naive Sobel\n";
    std::cout << "Image  : " << w << "x" << h << '\n';
    std::cout << "Warmup : complete\n";
    std::cout << "Runtime: " << ms << " ms\n\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void sobelExample(const char* imagePath, bool benchmark) {
    int w, h;
    int channels = 1;

    unsigned char* image = stbi_load(imagePath, &w, &h, &channels, 1);

    if (!image) {
        std::cerr << "Failed to load image: "
                  << imagePath << '\n';
        return;
    }

    const int pixelCount = w * h;
    const size_t imageBytes = pixelCount;

    unsigned char* sobelImage = nullptr;
    unsigned char* devInput = nullptr;
    unsigned char* devOutput = nullptr;

    CUDA_CHECK(cudaMallocHost(&sobelImage, imageBytes));

    CUDA_CHECK(cudaMalloc(&devInput, imageBytes));
    CUDA_CHECK(cudaMalloc(&devOutput, imageBytes));

    CUDA_CHECK(cudaMemcpy(devInput,
                          image,
                          imageBytes,
                          cudaMemcpyHostToDevice));

    if (benchmark)
    {
        benchmarkSobelKernel(devInput,
                            devOutput,
                            w,
                            h);
    }
    else
    {
        constexpr int threads = 256;
        const int blocks = (pixelCount + threads - 1) / threads;

        sobelKernel<<<blocks, threads>>>(devInput,
                                        devOutput,
                                        w,
                                        h);

        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(sobelImage, devOutput, imageBytes, cudaMemcpyDeviceToHost));

    stbi_write_jpg("output/sobel.jpg", w, h, 1, sobelImage, 100);

    CUDA_CHECK(cudaFree(devInput));
    CUDA_CHECK(cudaFree(devOutput));
    CUDA_CHECK(cudaFreeHost(sobelImage));

    stbi_image_free(image);
}
