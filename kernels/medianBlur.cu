#include <cuda_runtime_api.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <iostream>

#include "cuda_check.h"
#include "stb_image.h"
#include "stb_image_write.h"

__global__ void mBlurKernel(unsigned char* devInput,
                            unsigned char* devOutput,
                            int w,
                            int h,
                            int channels,
                            int mKernel) {

    // 1. Median blur
    // 2. Variable window sizes
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelCount = w * h;
    
    if (pixelIdx >= pixelCount) return;

    int rgbIdx = pixelIdx * channels;
    
    int x = pixelIdx % w;
    int y = pixelIdx / w;

    int radius = mKernel / 2;

    if (x < radius || x >= w - radius ||
        y < radius || y >= h - radius) {
            
            devOutput[rgbIdx + 0] = devInput[rgbIdx + 0];
            devOutput[rgbIdx + 1] = devInput[rgbIdx + 1];
            devOutput[rgbIdx + 2] = devInput[rgbIdx + 2];
            return;
        }

    unsigned char rWindow[225];
    unsigned char gWindow[225];
    unsigned char bWindow[225];

    int count = 0;
    
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int neighborX = x + dx;
            int neighborY = y + dy;

            int neighborIdx = neighborY * w + neighborX;
            int neighborRgbIdx = neighborIdx * channels;

            rWindow[count] = devInput[neighborRgbIdx + 0];
            gWindow[count] = devInput[neighborRgbIdx + 1];
            bWindow[count] = devInput[neighborRgbIdx + 2];

            count++;
        }
    }

    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (rWindow[i] > rWindow[j]) {
                unsigned char temp = rWindow[i];
                rWindow[i] = rWindow[j];
                rWindow[j] = temp;
            }
        }
    }
        
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (gWindow[i] > gWindow[j]) {
                unsigned char temp = gWindow[i];
                gWindow[i] = gWindow[j];
                gWindow[j] = temp;
            }
        }
    }

    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (bWindow[i] > bWindow[j]) {
                unsigned char temp = bWindow[i];
                bWindow[i] = bWindow[j];
                bWindow[j] = temp;
            }
        }
    }

    int medianIdx = count / 2;

    unsigned char rMedian = rWindow[medianIdx];
    unsigned char gMedian = gWindow[medianIdx];
    unsigned char bMedian = bWindow[medianIdx];

    devOutput[rgbIdx + 0] = rMedian;
    devOutput[rgbIdx + 1] = gMedian;
    devOutput[rgbIdx + 2] = bMedian;
}

void benchmarkMedianBlurKernel(unsigned char* devInput,
                               unsigned char* devOutput,
                               int w,
                               int h,
                               int channels,
                               int kernelSize) {

    constexpr int threads = 256;

    const int pixelCount = w * h;
    const int blocks = (pixelCount + threads - 1) / threads;

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup launch
    mBlurKernel<<<blocks, threads>>>(devInput,
                                    devOutput,
                                    w,
                                    h,
                                    channels,
                                    kernelSize);

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    mBlurKernel<<<blocks, threads>>>(devInput,
                                    devOutput,
                                    w,
                                    h,
                                    channels,
                                    kernelSize);

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "\nMedian Blur\n";
    std::cout << "Image  : " << w << "x" << h << '\n';
    std::cout << "Window : " << kernelSize << "x" << kernelSize << '\n';
    std::cout << "Warmup : complete\n";
    std::cout << "Runtime: " << ms << " ms\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void medianBlurExample(const char* imagePath, int kernelSize, bool benchmark) {
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
    const size_t blurBytes = pixelCount * channels;

    unsigned char* blurredImage = nullptr;
    unsigned char* devInput = nullptr;
    unsigned char* devOutput = nullptr;

    CUDA_CHECK(cudaMallocHost(&blurredImage, blurBytes));

    CUDA_CHECK(cudaMalloc(&devInput, rgbBytes));
    CUDA_CHECK(cudaMalloc(&devOutput, blurBytes));

    CUDA_CHECK(cudaMemcpy(devInput, image, rgbBytes, cudaMemcpyHostToDevice));

    if (benchmark) {
        benchmarkMedianBlurKernel(devInput,
                                devOutput,
                                w,
                                h,
                                channels,
                                kernelSize);
    } else {
        constexpr int threads = 256;
        const int blocks = (pixelCount + threads - 1) / threads;

        mBlurKernel<<<blocks, threads>>>(devInput,
                                        devOutput,
                                        w,
                                        h,
                                        channels,
                                        kernelSize);

        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(blurredImage, devOutput, blurBytes, cudaMemcpyDeviceToHost));

    stbi_write_jpg("output/medianBlur.jpg", w, h, channels, blurredImage, 100);

    CUDA_CHECK(cudaFree(devInput));
    CUDA_CHECK(cudaFree(devOutput));
    CUDA_CHECK(cudaFreeHost(blurredImage));

    stbi_image_free(image);
}
