// g++ -O3 -march=native -funroll-all-loops -mavx2 -o dot dot.cpp

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>

#define VECTOR_SIZE 104857600
#define NUM_RUNS 100
#define BLOCK_NUM 2048

using namespace std;


int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1, 1);

    // block mem malloc
    vector<float*> vectorA(BLOCK_NUM);
    vector<float*> vectorB(BLOCK_NUM);
    for(int i = 0; i < BLOCK_NUM; i++){
        vectorA[i] = static_cast<float*>(_mm_malloc((VECTOR_SIZE / BLOCK_NUM) * sizeof(float), 32));
        vectorB[i] = static_cast<float*>(_mm_malloc((VECTOR_SIZE / BLOCK_NUM) * sizeof(float), 32));
    }
    // random
    for (int i = 0; i < BLOCK_NUM; i++) {
        for (int j = 0; j < VECTOR_SIZE / BLOCK_NUM; j++){
            vectorA[i][j] = dis(gen);
            vectorB[i][j] = dis(gen);
        }
    }

    double dot_product;
    clock_t start, end;
    double total_time = 0.0;
    for (int run = 0; run < NUM_RUNS; run++) {
        __m256 sum = _mm256_setzero_ps();
        __m128 sum_high;
        __m128 sum_low;
        int i, j;

        auto start = std::chrono::high_resolution_clock::now();

        for(i = 0; i < BLOCK_NUM; i++){
            _mm_prefetch((const char*)(vectorA[i]), _MM_HINT_T0);
            for (j = 0; j <= (VECTOR_SIZE / BLOCK_NUM) - 56; j += 56) {
                __asm__ volatile (
                    "vmovaps (%0), %%ymm0\n"
                    "vmovaps 0x20(%0), %%ymm2\n"
                    "vmovaps 0x40(%0), %%ymm4\n"
                    "vmovaps 0x60(%0), %%ymm6\n"
                    "vmovaps 0x80(%0), %%ymm8\n"
                    "vmovaps 0xa0(%0), %%ymm10\n"
                    "vmovaps 0xc0(%0), %%ymm12\n"

                    "vmovaps (%1), %%ymm1\n"
                    "vmovaps 0x20(%1), %%ymm3\n"
                    "vmovaps 0x40(%1), %%ymm5\n"
                    "vmovaps 0x60(%1), %%ymm7\n"
                    "vmovaps 0x80(%1), %%ymm9\n"
                    "vmovaps 0xa0(%1), %%ymm11\n"
                    "vmovaps 0xc0(%1), %%ymm13\n"

                    "vfmadd231ps %%ymm0, %%ymm1, %%ymm14\n"
                    "vfmadd231ps %%ymm2, %%ymm3, %%ymm14\n"
                    "vfmadd231ps %%ymm4, %%ymm5, %%ymm14\n"
                    "vfmadd231ps %%ymm6, %%ymm7, %%ymm14\n"
                    "vfmadd231ps %%ymm8, %%ymm9, %%ymm14\n"
                    "vfmadd231ps %%ymm10, %%ymm11, %%ymm14\n"
                    "vfmadd231ps %%ymm12, %%ymm13, %%ymm14\n"

                    : // No output operands
                    : "r" (vectorA[i] + j), "r" (vectorB[i] + j)
                    : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14"
                );
            }
            __asm__ volatile (
                    "vmovaps (%0), %%ymm0\n"
                    "vmovaps 0x20(%0), %%ymm2\n"

                    "vmovaps (%1), %%ymm1\n"
                    "vmovaps 0x20(%1), %%ymm3\n"

                    "vfmadd231ps %%ymm0, %%ymm1, %%ymm14\n"
                    "vfmadd231ps %%ymm2, %%ymm3, %%ymm14\n"

                    : // No output operands
                    : "r" (vectorA[i] + j), "r" (vectorB[i] + j)
                    : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm14"
            );
        }
        __asm__ volatile (
            "vmovaps %%ymm14, %0"
            : "=m" (sum)  // output: sum
            : // No input operands
            : "ymm14" // Clobbered register
        );
        sum_high = _mm256_extractf128_ps(sum, 1);
        sum_low = _mm256_extractf128_ps(sum, 0);
        sum_low = _mm_add_ps(sum_low, sum_high);
        sum_low = _mm_hadd_ps(sum_low, sum_low);
        sum_low = _mm_hadd_ps(sum_low, sum_low);
        dot_product = _mm_cvtss_f32(sum_low);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << duration.count() << "s" << std::endl;
    }

    printf("Dot product: %f\n", dot_product);

    return 0;
}