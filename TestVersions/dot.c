// gcc -O3 -march=native -funroll-all-loops -mavx2 -o dot dot.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

#define VECTOR_SIZE 104857600
#define NUM_RUNS 200


typedef float FLOAT;

#if defined(_WIN64) || defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__)
    typedef intptr_t BLASLONG;
#else
    typedef long BLASLONG;
#endif


#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif

int main() {
    srand(time(NULL));

    const BLASLONG vector_len = VECTOR_SIZE;

    
    if (vector_len % 8 != 0) {
        fprintf(stderr, "Error: VECTOR_SIZE (%lld) must be divisible by 8 for AVX operations.\n", vector_len);
        return 1;
    }
    
    
    FLOAT *vectorA = (FLOAT *)_mm_malloc((size_t)vector_len * sizeof(FLOAT), 32);
    FLOAT *vectorB = (FLOAT *)_mm_malloc((size_t)vector_len * sizeof(FLOAT), 32);
    if (!vectorA || !vectorB) {
        perror("Failed to allocate aligned memory");
        if (vectorA) _mm_free(vectorA);
        if (vectorB) _mm_free(vectorB);
        return 1;
    }

    
    printf("Initializing vectors (size: %lld)...\n", vector_len);
    for (BLASLONG j = 0; j < vector_len; j++) {
        vectorA[j] = ((float)rand() / (float)(RAND_MAX)) * 2.0f - 1.0f;
        vectorB[j] = ((float)rand() / (float)(RAND_MAX)) * 2.0f - 1.0f;
    }

    double *runtimes_ms = (double *)malloc(NUM_RUNS * sizeof(double));
     if (!runtimes_ms) {
         perror("Failed to allocate runtimes array");
         _mm_free(vectorA);
         _mm_free(vectorB);
         return 1;
    }


    struct timespec start_time_ts, end_time_ts;
    double final_dot_product = 0.0;


    printf("Running benchmark (%d runs)...\n", NUM_RUNS);
    for (int run = 0; run < NUM_RUNS; run++) {

        clock_gettime(CLOCK_MONOTONIC, &start_time_ts);
        

        __m256 sum_vec0 = _mm256_setzero_ps();
        __m256 sum_vec1 = _mm256_setzero_ps();
        __m256 sum_vec2 = _mm256_setzero_ps();
        __m256 sum_vec3 = _mm256_setzero_ps();
        __m256 sum_vec4 = _mm256_setzero_ps();
        __m256 sum_vec5 = _mm256_setzero_ps();
        __m256 sum_vec6 = _mm256_setzero_ps();
        __m256 sum_vec7 = _mm256_setzero_ps();

        FLOAT *ptrA = vectorA;
        FLOAT *ptrB = vectorB;
        BLASLONG n = vector_len;
        BLASLONG j = 0;

        
        BLASLONG n_loops_64 = n / 64;

        if (n_loops_64 > 0) {
            BLASLONG loop_idx = 0;
            BLASLONG n_actual_loops = n_loops_64;

            __asm__  __volatile__
            (
                
                ".p2align 5                                    \n\t"
                "1:                                            \n\t"

                
                
                
                "vmovaps                  (%[a],%[li],4), %%ymm8          \n\t"
                "vmovaps                32(%[a],%[li],4), %%ymm9          \n\t"
                "vmovaps                64(%[a],%[li],4), %%ymm10         \n\t"
                "vmovaps                96(%[a],%[li],4), %%ymm11         \n\t"
                "vmovaps               128(%[a],%[li],4), %%ymm12         \n\t"
                "vmovaps               160(%[a],%[li],4), %%ymm13         \n\t"
                "vmovaps               192(%[a],%[li],4), %%ymm14         \n\t"
                "vmovaps               224(%[a],%[li],4), %%ymm15         \n\t"

                
                
                "vfmadd231ps      (%[b],%[li],4), %%ymm8 , %[s0] \n\t"
                "vfmadd231ps    32(%[b],%[li],4), %%ymm9 , %[s1] \n\t"
                "vfmadd231ps    64(%[b],%[li],4), %%ymm10, %[s2] \n\t"
                "vfmadd231ps    96(%[b],%[li],4), %%ymm11, %[s3] \n\t"
                "vfmadd231ps   128(%[b],%[li],4), %%ymm12, %[s4] \n\t"
                "vfmadd231ps   160(%[b],%[li],4), %%ymm13, %[s5] \n\t"
                "vfmadd231ps   192(%[b],%[li],4), %%ymm14, %[s6] \n\t"
                "vfmadd231ps   224(%[b],%[li],4), %%ymm15, %[s7] \n\t"


                
                "addq       $64 , %[li]                   \n\t"
                "subq       $1  , %[nl]                   \n\t"
                "jnz        1b                            \n\t"

                
                : [li] "+r" (loop_idx),
                  [nl] "+r" (n_actual_loops),
                  [s0] "+v" (sum_vec0), [s1] "+v" (sum_vec1),
                  [s2] "+v" (sum_vec2), [s3] "+v" (sum_vec3),
                  [s4] "+v" (sum_vec4), [s5] "+v" (sum_vec5),
                  [s6] "+v" (sum_vec6), [s7] "+v" (sum_vec7)
                
                : [a] "r" (ptrA),
                  [b] "r" (ptrB)
                
                : "memory", "cc",
                  
                  "ymm8", "ymm9", "ymm10", "ymm11",
                  "ymm12", "ymm13", "ymm14", "ymm15"
            );
            j = loop_idx;
        } 


        sum_vec0 = _mm256_add_ps(sum_vec0, sum_vec4);
        sum_vec1 = _mm256_add_ps(sum_vec1, sum_vec5);
        sum_vec2 = _mm256_add_ps(sum_vec2, sum_vec6);
        sum_vec3 = _mm256_add_ps(sum_vec3, sum_vec7);

        sum_vec0 = _mm256_add_ps(sum_vec0, sum_vec2);
        sum_vec1 = _mm256_add_ps(sum_vec1, sum_vec3);

        sum_vec0 = _mm256_add_ps(sum_vec0, sum_vec1);
        
        
        BLASLONG remaining = n - j;
        while (remaining >= 8) {
            __m256 a_vec = _mm256_load_ps(ptrA + j);
            __m256 b_vec = _mm256_load_ps(ptrB + j);
            
            
            sum_vec0 = _mm256_fmadd_ps(a_vec, b_vec, sum_vec0);
            j += 8;
            remaining -= 8;
        }

        __m128 sum_high = _mm256_extractf128_ps(sum_vec0, 1);
        __m128 sum_low  = _mm256_castps256_ps128(sum_vec0);
        
        sum_low = _mm_add_ps(sum_low, sum_high);
        
        sum_low = _mm_hadd_ps(sum_low, sum_low);
        sum_low = _mm_hadd_ps(sum_low, sum_low);

        
        float vector_dot_product_f = _mm_cvtss_f32(sum_low);
        final_dot_product = (double)vector_dot_product_f;


        double scalar_sum = 0.0;
        while (remaining > 0) {
            scalar_sum += (double)ptrA[j] * (double)ptrB[j];
            j++;
            remaining--;
        }
        final_dot_product += scalar_sum;

        clock_gettime(CLOCK_MONOTONIC, &end_time_ts);

        
        double start_ms = (double)start_time_ts.tv_sec * 1000.0 + (double)start_time_ts.tv_nsec / 1000000.0;
        double end_ms   = (double)end_time_ts.tv_sec * 1000.0 + (double)end_time_ts.tv_nsec / 1000000.0;
        runtimes_ms[run] = end_ms - start_ms;

    }
    
    printf("Dot product (last run): %f\n", final_dot_product);

    double time_sum_ms = 0.0;
    double min_time_ms = (NUM_RUNS > 0) ? runtimes_ms[0] : 0.0;
    double max_time_ms = (NUM_RUNS > 0) ? runtimes_ms[0] : 0.0;

    for (int run = 0; run < NUM_RUNS; run++) {
        time_sum_ms += runtimes_ms[run];
        if (runtimes_ms[run] < min_time_ms) {
            min_time_ms = runtimes_ms[run];
        }
        if (runtimes_ms[run] > max_time_ms) {
            max_time_ms = runtimes_ms[run];
        }
    }

    double avg_time_ms = (NUM_RUNS > 0) ? (time_sum_ms / NUM_RUNS) : 0.0;
    printf("Average time: %f ms\n", avg_time_ms);
    printf("Minimum time: %f ms\n", min_time_ms);
    printf("Maximum time: %f ms\n", max_time_ms);

    
    if (avg_time_ms > 0) {
        double avg_time_s = avg_time_ms / 1000.0;
        double total_flops = (double)vector_len * 2.0;
        double gflops = total_flops / (avg_time_s * 1e9);
        printf("Average Effective Computation GFLOPS: %f\n", gflops);
    }

    _mm_free(vectorA);
    _mm_free(vectorB);
    free(runtimes_ms);

    return 0;
}