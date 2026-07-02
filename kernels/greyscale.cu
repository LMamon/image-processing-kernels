#include <cuda_runtime_api.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <iostream>

#include "cuda_check.h"
#include "stb_image.h"
#include "stb_image_write.h"

__global__ void greyscaleKernel(unsigned char* rgb, unsigned char* grey, int width, int height, int channels) {
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelCount = width * height;

    if (pixelIdx >= pixelCount)
        return;

    int rgbIdx = pixelIdx * channels;
    
    unsigned char r = rgb[rgbIdx + 0];
    unsigned char g = rgb[rgbIdx + 1];
    unsigned char b = rgb[rgbIdx + 2];

    grey[pixelIdx] = 
        static_cast<unsigned char>(
            0.299f * r +
            0.587f * g +
            0.114f * b);
}

void benchmarkGreyscaleKernel(unsigned char* devInput,
                              unsigned char* devOutput,
                              int width,
                              int height,
                              int channels) {

    constexpr int threads = 256;

    const int pixelCount = width * height;
    const int blocks = (pixelCount + threads - 1) / threads;

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup launch
    greyscaleKernel<<<blocks, threads>>>(devInput,
                                        devOutput,
                                        width,
                                        height,
                                        channels);

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));

    greyscaleKernel<<<blocks, threads>>>(devInput,
                                        devOutput,
                                        width,
                                        height,
                                        channels);

    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "\nGreyscale\n";
    std::cout << "Image  : " << width << "x" << height << '\n';
    std::cout << "Warmup : complete\n";
    std::cout << "Runtime: " << ms << " ms\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void greyscaleExample(const char* imagePath, bool benchmark) {
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
    const size_t greyBytes = pixelCount;

    unsigned char* greyImage = nullptr;

    unsigned char* devRgb = nullptr;
    unsigned char* devGrey = nullptr;

    CUDA_CHECK(cudaMallocHost(&greyImage, greyBytes));

    CUDA_CHECK(cudaMalloc(&devRgb, rgbBytes));
    CUDA_CHECK(cudaMalloc(&devGrey, greyBytes));

    CUDA_CHECK(cudaMemcpy(devRgb, image, rgbBytes, cudaMemcpyHostToDevice));

    if (benchmark) {
        benchmarkGreyscaleKernel(devRgb,
                                devGrey,
                                w,
                                h,
                                channels);
    } else {
        constexpr int threads = 256;

        const int blocks = (pixelCount + threads - 1) / threads;

        greyscaleKernel<<<blocks, threads>>>(devRgb,
                                            devGrey,
                                            w,
                                            h,
                                            channels);

        CUDA_KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(greyImage, devGrey, greyBytes, cudaMemcpyDeviceToHost));

    stbi_write_jpg("output/greyscale.jpg", w, h, 1, greyImage, 100);

    CUDA_CHECK(cudaFree(devRgb));
    CUDA_CHECK(cudaFree(devGrey));
    CUDA_CHECK(cudaFreeHost(greyImage));

    stbi_image_free(image);
}
