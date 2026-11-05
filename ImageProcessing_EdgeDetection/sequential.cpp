#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <iomanip>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace std::chrono;

struct ResolutionTest {
    string label;
    string inputFile;
    string outputGrayFile;
    string outputSobelFile;
};

// ----------------------------------------------------------------------
// Section 3.0 (Methodology): Sobel Edge Detection Algorithm
// ----------------------------------------------------------------------
void applySobelSequential(const unsigned char* I, unsigned char* G, int width, int height) {
    // 3.1.2 Matrix Representation: 3x3 Horizontal (Kx) and Vertical (Ky) Kernels
    const int Kx[3][3] = {
        { -1,  0,  1 },
        { -2,  0,  2 },
        { -1,  0,  1 }
    };

    const int Ky[3][3] = {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };

    // Traverse interior pixels (skipping 1-pixel outermost border)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int Gx = 0;
            int Gy = 0;

            // 3.1.1 Scalar Form: 3x3 neighborhood summation
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    int neighbor_pixel = I[(y + j) * width + (x + i)];

                    Gx += neighbor_pixel * Kx[j + 1][i + 1];
                    Gy += neighbor_pixel * Ky[j + 1][i + 1];
                }
            }

            // Calculate gradient magnitude: sqrt(Gx^2 + Gy^2)
            int magnitude = static_cast<int>(sqrt(Gx * Gx + Gy * Gy));

            // Clamp value to maximum 255 and store in output matrix G
            G[y * width + x] = static_cast<unsigned char>(min(255, magnitude));
        }
    }
}

// Helper function to run and time a single image
bool processImage(const ResolutionTest& test) {
    int width, height, channels;
    unsigned char* input_I = stbi_load(test.inputFile.c_str(), &width, &height, &channels, 1);

    if (!input_I) {
        cerr << "[Error] Could not load '" << test.inputFile << "'. Ensure it is in the project folder.\n";
        return false;
    }

    // Save grayscale image with resolution-specific name
    stbi_write_png(test.outputGrayFile.c_str(), width, height, 1, input_I, width);

    // Allocate output buffer G initialized to 0 (black)
    vector<unsigned char> output_G(width * height, 0);

    // ================= START BENCHMARK TIMER =================
    auto start = high_resolution_clock::now();

    applySobelSequential(input_I, output_G.data(), width, height);

    auto end = high_resolution_clock::now();
    duration<double, milli> elapsed = end - start;
    // ================== END BENCHMARK TIMER ==================

    // Save edge-detected image with resolution-specific name
    stbi_write_png(test.outputSobelFile.c_str(), width, height, 1, output_G.data(), width);

    cout << left << setw(22) << test.label
        << setw(14) << (to_string(width) + "x" + to_string(height))
        << fixed << setprecision(3) << setw(12) << elapsed.count() << " ms"
        << " -> Saved: " << test.outputSobelFile << "\n";

    stbi_image_free(input_I);
    return true;
}

int main() {
    const char* inputFiles[5] = { "720p.jpg", "1080p.jpg", "1440p.jpg", "1800p.jpg", "2160p.jpg" };
    const char* grayFiles[5] = { "gray_720p.png", "gray_1080p.png", "gray_1440p.png", "gray_1800p.png", "gray_2160p.png" };
    const char* sobelFiles[5] = { "sobel_seq_720p.png", "sobel_seq_1080p.png", "sobel_seq_1440p.png", "sobel_seq_1800p.png", "sobel_seq_2160p.png" };

    cout << "Select image resolution to test:\n"
        << "1. 1280 x 720  (720p.jpg)\n"
        << "2. 1920 x 1080 (1080p.jpg)\n"
        << "3. 2560 x 1440 (1440p.jpg)\n"
        << "4. 3200 x 1800 (1800p.jpg)\n"
        << "5. 3840 x 2160 (2160p.jpg)\n"
        << "Enter choice (1-5): ";

    int choice;
    cin >> choice;

    if (choice < 1 || choice > 5) {
        cerr << "Invalid choice!\n";
        return 1;
    }

    const char* filename = inputFiles[choice - 1];
    const char* grayOut = grayFiles[choice - 1];
    const char* sobelOut = sobelFiles[choice - 1];

    int width, height, channels;
    unsigned char* input_I = stbi_load(filename, &width, &height, &channels, 1);
    if (!input_I) {
        cerr << "Error: Could not load image '" << filename << "'! Make sure it is in the same folder.\n";
        return 1;
    }

    cout << "\nSuccessfully loaded image: " << filename << " (" << width << "x" << height << " pixels)\n";

    stbi_write_png(grayOut, width, height, 1, input_I, width);

    vector<unsigned char> output_G(width * height, 0);

    // ================= START BENCHMARK TIMER =================
    auto start = high_resolution_clock::now();

    applySobelSequential(input_I, output_G.data(), width, height);

    auto end = high_resolution_clock::now();
    duration<double, milli> elapsed = end - start;
    // ================== END BENCHMARK TIMER ==================

    cout << "Sequential Execution Time (Ts): " << elapsed.count() << " ms\n";

    stbi_write_png(sobelOut, width, height, 1, output_G.data(), width);
    cout << "Saved edge-detected image to '" << sobelOut << "'\n";

    stbi_image_free(input_I);
    return 0;
}