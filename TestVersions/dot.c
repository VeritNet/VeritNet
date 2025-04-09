// gcc -O3 -march=native -funroll-all-loops -mavx2 -o dot dot.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <time.h>      
#include <immintrin.h> 
#include <math.h>      
#include <float.h>     

#define VECTOR_SIZE 104857600
#define NUM_RUNS 20
#define BLOCK_NUM 1 


#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif


inline double timespec_diff_ms(struct timespec *start, struct timespec *end) {
    struct timespec temp;
    if ((end->tv_nsec - start->tv_nsec) < 0) {
        temp.tv_sec = end->tv_sec - start->tv_sec - 1;
        temp.tv_nsec = 1000000000 + end->tv_nsec - start->tv_nsec;
    } else {
        temp.tv_sec = end->tv_sec - start->tv_sec;
        temp.tv_nsec = end->tv_nsec - start->tv_nsec;
    }
    return (double)temp.tv_sec * 1000.0 + (double)temp.tv_nsec / 1000000.0;
}


int main() {
    
    srand(time(NULL));

    const int size_per_block = VECTOR_SIZE / BLOCK_NUM;

    
    if (size_per_block % 8 != 0) {
        fprintf(stderr, "Error: VECTOR_SIZE / BLOCK_NUM (%d) must be divisible by 8 for AVX operations.\n", size_per_block);
        return 1;
    }
     
     if (VECTOR_SIZE < 56 && VECTOR_SIZE != 16) {
         fprintf(stderr, "Warning: VECTOR_SIZE is too small for the main loop structure and the specific remainder handling.\n");
         
     }
    
    float **vectorA = (float **)malloc(BLOCK_NUM * sizeof(float *));
    float **vectorB = (float **)malloc(BLOCK_NUM * sizeof(float *));
    if (!vectorA || !vectorB) {
        perror("Failed to allocate block pointers");
        return 1;
    }
    
    for (int i = 0; i < BLOCK_NUM; i++) {
        vectorA[i] = (float *)_mm_malloc((size_t)size_per_block * sizeof(float), 32); 
        vectorB[i] = (float *)_mm_malloc((size_t)size_per_block * sizeof(float), 32); 
        if (!vectorA[i] || !vectorB[i]) {
            fprintf(stderr, "Failed to allocate aligned memory for block %d\n", i);
            
            for(int k=0; k<i; ++k) {
                 if(vectorA[k]) _mm_free(vectorA[k]);
                 if(vectorB[k]) _mm_free(vectorB[k]);
            }
            if(vectorA[i]) _mm_free(vectorA[i]); 
            free(vectorA);
            free(vectorB);
            return 1;
        }
    }

    
    for (int i = 0; i < BLOCK_NUM; i++) {
        for (int j = 0; j < size_per_block; j++) {
            
            vectorA[i][j] = ((float)rand() / (float)(RAND_MAX)) * 2.0f - 1.0f;
            vectorB[i][j] = ((float)rand() / (float)(RAND_MAX)) * 2.0f - 1.0f;
        }
    }

    double dot_product = 0.0; 
    double *runtimes = (double *)malloc(NUM_RUNS * sizeof(double));
    if (!runtimes) {
         perror("Failed to allocate runtimes array");
         
         for (int i = 0; i < BLOCK_NUM; i++) {
             _mm_free(vectorA[i]);
             _mm_free(vectorB[i]);
         }
         free(vectorA);
         free(vectorB);
         return 1;
    }


    struct timespec start_time_ts, end_time_ts; 

    for (int run = 0; run < NUM_RUNS; run++) {
        __m128 sum_high;
        __m128 sum_low;
        int i; 
        int j = 0; 

        clock_gettime(CLOCK_MONOTONIC, &start_time_ts);

        __m256 sum = _mm256_setzero_ps(); 

        for (i = 0; i < BLOCK_NUM; i++) {
            _mm_prefetch((const char*)(vectorA[i]), _MM_HINT_T0);
            _mm_prefetch((const char*)(vectorB[i]), _MM_HINT_T0);

            j = 0; 
            
            int limit = size_per_block - 56;

            for (; j <= limit; j += 56) { 
                __asm__ volatile (
                    
                    "vmovaps (%1), %%ymm0\n"
                    "vmovaps 0x20(%1), %%ymm2\n"
                    "vmovaps 0x40(%1), %%ymm4\n"
                    "vmovaps 0x60(%1), %%ymm6\n"
                    "vmovaps 0x80(%1), %%ymm8\n"
                    "vmovaps 0xa0(%1), %%ymm10\n"
                    "vmovaps 0xc0(%1), %%ymm12\n"

                    
                    "vmovaps (%2), %%ymm1\n"
                    "vmovaps 0x20(%2), %%ymm3\n"
                    "vmovaps 0x40(%2), %%ymm5\n"
                    "vmovaps 0x60(%2), %%ymm7\n"
                    "vmovaps 0x80(%2), %%ymm9\n"
                    "vmovaps 0xa0(%2), %%ymm11\n"
                    "vmovaps 0xc0(%2), %%ymm13\n"

                    
                    "vfmadd231ps %%ymm0, %%ymm1, %0\n" 
                    "vfmadd231ps %%ymm2, %%ymm3, %0\n"
                    "vfmadd231ps %%ymm4, %%ymm5, %0\n"
                    "vfmadd231ps %%ymm6, %%ymm7, %0\n"
                    "vfmadd231ps %%ymm8, %%ymm9, %0\n"
                    "vfmadd231ps %%ymm10, %%ymm11, %0\n"
                    "vfmadd231ps %%ymm12, %%ymm13, %0\n"

                    : "+x" (sum) 
                    : "r" (vectorA[i] + j), "r" (vectorB[i] + j) 
                    : "memory", 
                      
                      "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7",
                      "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13"
                );
            } 
            
            if (size_per_block - j == 16) {
                 __asm__ volatile (
                    
                    "vmovaps (%1), %%ymm0\n"
                    "vmovaps 0x20(%1), %%ymm2\n"

                    
                    "vmovaps (%2), %%ymm1\n"
                    "vmovaps 0x20(%2), %%ymm3\n"

                    
                    "vfmadd231ps %%ymm0, %%ymm1, %0\n"
                    "vfmadd231ps %%ymm2, %%ymm3, %0\n"

                    : "+x" (sum)
                    : "r" (vectorA[i] + j), "r" (vectorB[i] + j)
                    : "memory", "ymm0", "ymm1", "ymm2", "ymm3"
                 );
                 
            }
        } 

        
        sum_high = _mm256_extractf128_ps(sum, 1); 
        sum_low = _mm256_extractf128_ps(sum, 0);  
        sum_low = _mm_add_ps(sum_low, sum_high);   
        
        
        sum_low = _mm_hadd_ps(sum_low, sum_low);
        
        sum_low = _mm_hadd_ps(sum_low, sum_low);
        
        dot_product = (double)_mm_cvtss_f32(sum_low); 

        clock_gettime(CLOCK_MONOTONIC, &end_time_ts);
        runtimes[run] = timespec_diff_ms(&start_time_ts, &end_time_ts);
    } 

    
    printf("Dot product: %f\n", dot_product);

    
    double time_sum = 0.0;
    
    double min_time = (NUM_RUNS > 0) ? runtimes[0] : 0.0;
    double max_time = (NUM_RUNS > 0) ? runtimes[0] : 0.0;

    for (int run = 0; run < NUM_RUNS; run++) {
        time_sum += runtimes[run];
        if (runtimes[run] < min_time) {
            min_time = runtimes[run];
        }
        if (runtimes[run] > max_time) {
            max_time = runtimes[run];
        }
    }

    double avg_time = (NUM_RUNS > 0) ? (time_sum / NUM_RUNS) : 0.0;
    printf("Average time: %f ms\n", avg_time);
    printf("Minimum time: %f ms\n", min_time);
    printf("Maximum time: %f ms\n", max_time);

    
    for (int i = 0; i < BLOCK_NUM; i++) {
        _mm_free(vectorA[i]);
        _mm_free(vectorB[i]);
    }
    free(vectorA);
    free(vectorB);
    free(runtimes);

    return 0;
}