#include <cuda_runtime_api.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <iostream>

#include "cuda_check.h"
#include "stb_image.h"
#include "stb_image_write.h"

__global__ void thresholdKernel(unsigned char* devInput, 
                                unsigned char* devOutput, 
                                int w, 
                                int h, 
                                int channels, 
                                int threshold) {

    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelCount = w * h;
    
    if (pixelIdx >= pixelCount) return;

    int rgbIdx = pixelIdx * channels;

    unsigned char r = devInput[rgbIdx + 0];
    unsigned char g = devInput[rgbIdx + 1];
    unsigned char b = devInput[rgbIdx + 2];

    float intensity = 0.299f * r + 0.587f * g + 0.114f * b;

    devOutput[pixelIdx] = intensity > threshold ? 255 : 0;
}
// load jpeg

// decode to rgb buffer
// cudaMalloc
// cudaMemcpy
// launch greyscale kernel
// write greyscale jpeg

void benchmarkThresholdKernel(unsigned char* devInput,
                              unsigned char* devOutput,
                              int w,
                              int h,
                              int channels,
                              int threshold) {

    constexpr int threads = 256;
    const int pixelCount = w * h;
    const int blocks = (pixelCount + threads - 1) / threads;

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup launch
    thresholdKernel<<<blocks, threads>>>(devInput,
                                        devOutput,
                                        w,
                                        h,
                                        channels,
                                        threshold);

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    thresholdKernel<<<blocks, threads>>>(devInput,
                                        devOutput,
                                        w,
                                        h,
                                        channels,
                                        threshold);

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "\nKernel: Threshold\n";
    std::cout << "Image  : " << w << "x" << h << '\n';
    std::cout << "Warmup : complete\n";
    std::cout << "Runtime: " << ms << " ms\n\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void thresholdExample(const char* imagePath, int threshold, bool benchmark) {
    int w, h;
    int channels = 3;
    unsigned char* image = stbi_load(imagePath, &w, &h, &channels, 3);

    if (!image) {
        std::cerr << "Failed to load image: "
                  << imagePath << '\n';
        return;
    }

    const int pixelCount = w * h;
    const size_t rgbBytes = pixelCount * channels;
    const size_t thresholdBytes = pixelCount;

    unsigned char* thresholdedImage = nullptr;
    unsigned char* devInput = nullptr;
    unsigned char* devOutput = nullptr;

    CUDA_CHECK(cudaMallocHost(&thresholdedImage, thresholdBytes));

    CUDA_CHECK(cudaMalloc(&devInput, rgbBytes));
    CUDA_CHECK(cudaMalloc(&devOutput, thresholdBytes));

    CUDA_CHECK(cudaMemcpy(devInput, image, rgbBytes, cudaMemcpyHostToDevice));

    if (benchmark) {
        benchmarkThresholdKernel(devInput,
                                devOutput,
                                w,
                                h,
                                channels,
                                threshold);
    } else {
        constexpr int threads = 256;
        const int blocks = (pixelCount + threads - 1) / threads;

        thresholdKernel<<<blocks, threads>>>(devInput,
                                            devOutput,
                                            w,
                                            h,
                                            channels,
                                            threshold);

        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(thresholdedImage, devOutput, thresholdBytes, cudaMemcpyDeviceToHost));

    stbi_write_jpg("output/threshold.jpg", w, h, 1, thresholdedImage, 100);

    CUDA_CHECK(cudaFree(devInput));
    CUDA_CHECK(cudaFree(devOutput));
    CUDA_CHECK(cudaFreeHost(thresholdedImage));

    stbi_image_free(image);
}
