#include <iostream>
#include <string>

#include "greyscale.cuh"
#include "threshold.cuh"
#include "sobel.cuh"
#include "smSobel.cuh"
#include "medianBlur.cuh"

#define CUDA_EXPRESSION_CHECKER
#include "cuda_check.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage:\n";
        std::cout << "  ./image-kernels <kernel> <image> [options]\n\n";

        std::cout << "Examples:\n";
        std::cout << "  ./image-kernels grayscale images/lena.jpg\n";
        std::cout << "  ./image-kernels threshold images/lena.jpg --threshold 150\n";
        std::cout << "  ./image-kernels smsobel images/lena.jpg --benchmark\n";

        return 1;
    }

    std::string kernel = argv[1];
    const char* imagePath = argv[2];

    bool benchmark = false;
    int kernelSize = 3;
    int threshold = 128;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--benchmark" || arg == "-b") {
            benchmark = true;
        }
        else if (arg == "--threshold") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after --threshold\n";
                return 1;
            }

            threshold = std::stoi(argv[++i]);
            if (threshold < 0 || threshold > 255) {
                std::cerr << "Threshold must be between 0 and 255.\n";
                return 1;
            }
        } 
        else if (arg == "--kernel-size" || arg == "-k") {
            if (i + 1 >= argc) {
                std::cerr << "Missing value after --kernel-size\n";
                return 1;
            }

            kernelSize = std::stoi(argv[++i]);

            if (kernelSize < 3 || kernelSize > 15 || kernelSize % 2 == 0) {
                std::cerr << "Kernel size must be an odd value 3-15.\n";
                return 1;
            }
        }
        else {
            std::cerr << "Unknown option: " << arg << '\n';
            return 1;
        }
    }

    if (kernel == "grayscale") {
        greyscaleExample(imagePath, benchmark);
    }
    else if (kernel == "threshold") {
        thresholdExample(imagePath, threshold, benchmark);
    }
    else if (kernel == "sobel") {
        sobelExample(imagePath, benchmark); }
    else if (kernel == "smsobel") {
        smSobelExample(imagePath, benchmark);
    }
    else if (kernel == "median") {
        medianBlurExample(imagePath, kernelSize, benchmark);
    }
    else {
        std::cerr << "Unknown kernel: " << kernel << '\n';
        return 1;
    }

    return 0;
}