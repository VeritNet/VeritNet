/*
* The codes are generated with Blueprint By VeritNet Engine, using AVX2, so I manually added comments and modified some variable names to make the codes easier to read.
* g++ -O3 -std=c++20 -march=native -funroll-all-loops -mavx2 -o v5.exe v5.cpp
* Version 2024.8.20.5
* [128 Elu, 32 Elu, 10 Softmax]
*/

#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <mutex>
#include <deque>
#include <chrono>
#include <atomic>
#include <semaphore>
#include <immintrin.h>

#include "MNIST.h"


using namespace std;


std::mutex mtx;//Global Mutex Lock
//Thread Pool
class TP {
public:
    vector<std::counting_semaphore<1>*> TGate;//Wake Conditon
    deque<atomic<bool>> TFree;//Is Free
    int TSize;//Num of threads
    std::vector<std::thread> init(int size);//Create & Detach
    inline void add() {//Add Task
        bool noBreak = true;
        while (noBreak) {
            for (int i = 0; i < TSize; i++) {//Find a Free Thread
                if (TFree[i].load()) {
                    TFree[i].store(false);
                    TGate[i]->release();
                    noBreak = false;
                    break;
                }
            }
        }
    }
};
TP* tpool = new TP;//thread pool
int dts;//Training Data Limit
int batchSize;
//Weight (and bias at the end of each array) (Shared Memory)
float* network0 = static_cast<float*>(_mm_malloc(128*784 * sizeof(float), 32));
float* network1 = static_cast<float*>(_mm_malloc(32*128 * sizeof(float), 32));
float* network2 = static_cast<float*>(_mm_malloc(10*32 * sizeof(float), 32));
float MSETotal;//MSE Cost
//Network Gradients (Shared Memory)
float* networkgs0 = static_cast<float*>(_mm_malloc(128*784 * sizeof(float), 32));
float* networkgs1 = static_cast<float*>(_mm_malloc(32*128 * sizeof(float), 32));
float* networkgs2 = static_cast<float*>(_mm_malloc(10*32 * sizeof(float), 32));

float network0_bi = 0.0f;
float network1_bi = 0.0f;
float network2_bi = 0.0f;
float networkgs0_bi = 0.0f;
float networkgs1_bi = 0.0f;
float networkgs2_bi = 0.0f;
//Locks for Blocks
std::vector<std::mutex> networkgs0_mtx(8);
std::vector<std::mutex> networkgs1_mtx(1);
std::vector<std::mutex> networkgs2_mtx(1);
bool gate;//Main Thread Gate
int reportI = 0;
int BId = 0;//Batch Id
float rate, aim, err;//Learning Rate, MSE aim, Cost of 1 Epoch
inline void trainNet(int TId/*Thread Id*/) {
    tpool->TGate[TId] = new std::counting_semaphore<1>(0);
    int thisDtId;

    //Block update of gradients in shared memory
    vector<bool> networkg0_todoList(8);
    vector<bool> networkg1_todoList(1);
    vector<bool> networkg2_todoList(1);
    
    float MSError{};//Error Sum of All Data in this Epoch in this Thread

    float SSum;//For Softmax Activation
    int i, p, q, dtIndex;//Loop Index
    int mtx_index_0, mtx_index_1, mtx_index_2;//Lock Index
    __m256 sum, factor;//For SIMD
    __m128 sum_high, sum_low;//For SIMD

    float* networkg0 = static_cast<float*>(_mm_malloc(128*784 * sizeof(float), 32));//Network Gradients (Thread Memory)
    float* networkb0 = static_cast<float*>(_mm_malloc(128 * sizeof(float), 32));//w*x + b (Before Activation) for hidden layer 0
    float* networkn0 = static_cast<float*>(_mm_malloc(128 * sizeof(float), 32));//Activation for hidden layer 0
    float* networkg1 = static_cast<float*>(_mm_malloc(32*128 * sizeof(float), 32));//Network Gradients (Thread Memory)
    float* networkb1 = static_cast<float*>(_mm_malloc(32 * sizeof(float), 32));//..for hidden layer 1
    float* networkn1 = static_cast<float*>(_mm_malloc(32 * sizeof(float), 32));//..for hidden layer 1
    float* networkg2 = static_cast<float*>(_mm_malloc(10*32 * sizeof(float), 32));//Network Gradients (Thread Memory)
    float* networkb2 = static_cast<float*>(_mm_malloc(10 * sizeof(float), 32));//..for hidden layer 2
    float* networkn2 = static_cast<float*>(_mm_malloc(10 * sizeof(float), 32));//..for hidden layer 2

    //bias gradiant
    float networkg0_bi = 0.0f;
    float networkg1_bi = 0.0f;
    float networkg2_bi = 0.0f;

    std::fill(networkg0, networkg0 + 100352, 0);
    std::fill(networkg1, networkg1 + 4096, 0);
    std::fill(networkg2, networkg2 + 320, 0);

    std::fill(networkn0, networkn0 + 128, 0);
    std::fill(networkn1, networkn1 + 32, 0);
    std::fill(networkn2, networkn2 + 10, 0);
    std::fill(networkb0, networkb0 + 128, 0);
    std::fill(networkb1, networkb1 + 32, 0);
    std::fill(networkb2, networkb2 + 10, 0);

    //The gradient of neurons is equal to the gradient of neuron bias
    float* networkg0_neuron = static_cast<float*>(_mm_malloc(128 * sizeof(float), 32));
    float* networkg1_neuron = static_cast<float*>(_mm_malloc(32 * sizeof(float), 32));
    float* networkg2_neuron = static_cast<float*>(_mm_malloc(10 * sizeof(float), 32));
    std::fill(networkg0_neuron, networkg0_neuron + 128, 0);
    std::fill(networkg1_neuron, networkg1_neuron + 32, 0);
    std::fill(networkg2_neuron, networkg2_neuron + 10, 0);


    for (;;) {
        tpool->TGate[TId]->acquire();
        for (dtIndex = batchSize / tpool->TSize - 1; dtIndex >= 0; dtIndex--) {//Train all data in this task
            thisDtId = (batchSize * BId) + (TId * (batchSize / tpool->TSize)) + dtIndex;
            //Feed Forward
            //Input Layer - Hidden Layer 0
            
            for (p = 0; p < 128; p++) {
                sum = _mm256_setzero_ps();
                i = 0;
                _mm_prefetch((const char*)(network0 + (p * 784)), _MM_HINT_T0);
                for (; i <= 784 - 64; i += 64) {
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i), _mm256_load_ps(train_image[thisDtId] + i), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 8), _mm256_load_ps(train_image[thisDtId] + i + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 16), _mm256_load_ps(train_image[thisDtId] + i + 16), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 24), _mm256_load_ps(train_image[thisDtId] + i + 24), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 32), _mm256_load_ps(train_image[thisDtId] + i + 32), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 40), _mm256_load_ps(train_image[thisDtId] + i + 40), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 48), _mm256_load_ps(train_image[thisDtId] + i + 48), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 56), _mm256_load_ps(train_image[thisDtId] + i + 56), sum);
                }
                for (; i <= 784 - 16; i += 16) {
                    _mm_prefetch((const char*)(network0 + i + 16), _MM_HINT_T0);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i), _mm256_load_ps(train_image[thisDtId] + i), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 8), _mm256_load_ps(train_image[thisDtId] + i + 8), sum);
                }
                sum_high = _mm256_extractf128_ps(sum, 1);
                sum_low = _mm256_extractf128_ps(sum, 0);
                sum_low = _mm_add_ps(sum_low, sum_high);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                networkb0[p] = _mm_cvtss_f32(sum_low);

                networkb0[p] += network0_bi;
                if (networkb0[p] >= 0) {
                    networkn0[p] = networkb0[p];
                } else {
                    networkn0[p] = exp(networkb0[p]) - 1;
                }
            }
            //Hidden Layer 0 - Hidden Layer 1
            for (p = 0; p < 32; p++) {
                sum = _mm256_setzero_ps();
                i = 0;
                _mm_prefetch((const char*)(network1), _MM_HINT_T0);
                for (; i <= 128 - 64; i += 64) {
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network1 + (p * 128) + i), _mm256_load_ps(networkn0 + i), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network1 + (p * 128) + i + 8), _mm256_load_ps(networkn0 + i + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network1 + (p * 128) + i + 16), _mm256_load_ps(networkn0 + i + 16), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network1 + (p * 128) + i + 24), _mm256_load_ps(networkn0 + i + 24), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network1 + (p * 128) + i + 32), _mm256_load_ps(networkn0 + i + 32), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network1 + (p * 128) + i + 40), _mm256_load_ps(networkn0 + i + 40), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network1 + (p * 128) + i + 48), _mm256_load_ps(networkn0 + i + 48), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network1 + (p * 128) + i + 56), _mm256_load_ps(networkn0 + i + 56), sum);
                }
                sum_high = _mm256_extractf128_ps(sum, 1);
                sum_low = _mm256_extractf128_ps(sum, 0);
                sum_low = _mm_add_ps(sum_low, sum_high);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                networkb1[p] = _mm_cvtss_f32(sum_low);

                networkb1[p] += network1_bi;
                if (networkb1[p] >= 0) {
                    networkn1[p] = networkb1[p];
                } else {
                    networkn1[p] = exp(networkb1[p]) - 1;
                }
            }
            //Hidden Layer 1 - Hidden Layer 2
            for (p = 0; p < 10; p++) {
                sum = _mm256_setzero_ps();
                i = 0;
                _mm_prefetch((const char*)(network2), _MM_HINT_T0);
                for (; i <= 32 - 32; i += 32) {
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network2 + (p * 32) + i), _mm256_load_ps(networkn1 + i), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network2 + (p * 32) + i + 8), _mm256_load_ps(networkn1 + i + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network2 + (p * 32) + i + 16), _mm256_load_ps(networkn1 + i + 16), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network2 + (p * 32) + i + 24), _mm256_load_ps(networkn1 + i + 24), sum);
                }
                sum_high = _mm256_extractf128_ps(sum, 1);
                sum_low = _mm256_extractf128_ps(sum, 0);
                sum_low = _mm_add_ps(sum_low, sum_high);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                networkb2[p] = _mm_cvtss_f32(sum_low);

                networkb2[p] += network2_bi;
            }
            //Hidden Layer 2 - Output Layer
            SSum = 0;
            for (i = 0; i < 10; i++) {
                SSum += exp(networkb2[i]);
            }
            for (i = 0; i < 10; i++) {
                networkn2[i] += exp(networkb2[i]) / SSum;
            }

            //Loss
            for (p = 0; p < 10; p++) {
                MSError += ((train_label[thisDtId][p] - networkn2[p]) * (train_label[thisDtId][p] - networkn2[p]));
            }

            //Back Propagation
            //Output Layer - Hidden Layer 2
            for (p = 0; p < 10; p++) {
                networkg2_neuron[p] = -rate * (networkn2[p] - train_label[thisDtId][p]);
                networkg2_bi += networkg2_neuron[p];
                _mm_prefetch((const char*)(networkg2 + (p * 32)), _MM_HINT_T0);
                __asm__ volatile (
                    // Load 4 ymm registers from a
                    "vmovaps (%0), %%ymm0\n"
                    "vmovaps 0x20(%0), %%ymm1\n"
                    "vmovaps 0x40(%0), %%ymm2\n"
                    "vmovaps 0x60(%0), %%ymm3\n"

                    // Load 4 ymm registers from b
                    "vmovaps (%1), %%ymm4\n"
                    "vmovaps 0x20(%1), %%ymm5\n"
                    "vmovaps 0x40(%1), %%ymm6\n"
                    "vmovaps 0x60(%1), %%ymm7\n"

                    // Load factor_vec into ymm15
                    "vmovaps %2, %%ymm15\n"

                    // Multiply a by factor and add to b using vfmadd231ps
                    "vfmadd231ps %%ymm15, %%ymm0, %%ymm4\n"
                    "vfmadd231ps %%ymm15, %%ymm1, %%ymm5\n"
                    "vfmadd231ps %%ymm15, %%ymm2, %%ymm6\n"
                    "vfmadd231ps %%ymm15, %%ymm3, %%ymm7\n"

                    // Store results back to b
                    "vmovaps %%ymm4, (%1)\n"
                    "vmovaps %%ymm5, 0x20(%1)\n"
                    "vmovaps %%ymm6, 0x40(%1)\n"
                    "vmovaps %%ymm7, 0x60(%1)\n"

                    : // No output operands
                    : "r" (networkn1), "r" (networkg2 + (p * 32)), "x" (_mm256_set1_ps(networkg2_neuron[p]))//Used only once: not set to factor
                    : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm15"
                );
            }
            //Hidden Layer 2 - Hidden Layer 1
            for (p = 0; p < 32; p++) {
                networkg1_neuron[p] = 0;
                for (q = 0; q < 10; q++) {
                    networkg1_neuron[p] += networkg2_neuron[q] * network2[(q * 32) + p];
                }
                if (networkb1[p] >= 0) {
                    networkg1_bi = networkg1_neuron[p];
                } else {
                    networkg1_neuron[p] *= exp(networkb1[p]);
                    networkg1_bi = networkg1_neuron[p];
                }
                _mm_prefetch((const char*)(networkg1 + (p * 128)), _MM_HINT_T0);
                factor = _mm256_set1_ps(networkg1_neuron[p]);
                for (i = 0; i <= 128 - 56; i += 56) {
                    __asm__ volatile (
                        // Load 7 ymm registers from a
                        "vmovaps (%0), %%ymm0\n"
                        "vmovaps 0x20(%0), %%ymm1\n"
                        "vmovaps 0x40(%0), %%ymm2\n"
                        "vmovaps 0x60(%0), %%ymm3\n"
                        "vmovaps 0x80(%0), %%ymm4\n"
                        "vmovaps 0xa0(%0), %%ymm5\n"
                        "vmovaps 0xc0(%0), %%ymm6\n"

                        // Load 7 ymm registers from b
                        "vmovaps (%1), %%ymm7\n"
                        "vmovaps 0x20(%1), %%ymm8\n"
                        "vmovaps 0x40(%1), %%ymm9\n"
                        "vmovaps 0x60(%1), %%ymm10\n"
                        "vmovaps 0x80(%1), %%ymm11\n"
                        "vmovaps 0xa0(%1), %%ymm12\n"
                        "vmovaps 0xc0(%1), %%ymm13\n"

                        // Load factor_vec into ymm14
                        "vmovaps %2, %%ymm14\n"

                        // Multiply a by factor and add to b using vfmadd231ps
                        "vfmadd231ps %%ymm14, %%ymm0, %%ymm7\n"
                        "vfmadd231ps %%ymm14, %%ymm1, %%ymm8\n"
                        "vfmadd231ps %%ymm14, %%ymm2, %%ymm9\n"
                        "vfmadd231ps %%ymm14, %%ymm3, %%ymm10\n"
                        "vfmadd231ps %%ymm14, %%ymm4, %%ymm11\n"
                        "vfmadd231ps %%ymm14, %%ymm5, %%ymm12\n"
                        "vfmadd231ps %%ymm14, %%ymm6, %%ymm13\n"

                        // Store results back to b
                        "vmovaps %%ymm7, (%1)\n"
                        "vmovaps %%ymm8, 0x20(%1)\n"
                        "vmovaps %%ymm9, 0x40(%1)\n"
                        "vmovaps %%ymm10, 0x60(%1)\n"
                        "vmovaps %%ymm11, 0x80(%1)\n"
                        "vmovaps %%ymm12, 0xa0(%1)\n"
                        "vmovaps %%ymm13, 0xc0(%1)\n"

                        : // No output operands
                        : "r" (networkn0 + i), "r" (networkg1 + (p * 128) + i), "x" (factor)
                        : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14"
                    );
                    __asm__ volatile (
                        // Load 2 ymm registers from a
                        "vmovaps (%0), %%ymm0\n"
                        "vmovaps 0x20(%0), %%ymm1\n"

                        // Load 2 ymm registers from b
                        "vmovaps (%1), %%ymm2\n"
                        "vmovaps 0x20(%1), %%ymm3\n"

                        // Load factor_vec into ymm15
                        "vmovaps %2, %%ymm15\n"

                        // Multiply a by factor and add to b using vfmadd231ps
                        "vfmadd231ps %%ymm15, %%ymm0, %%ymm2\n"
                        "vfmadd231ps %%ymm15, %%ymm1, %%ymm3\n"

                        // Store results back to b
                        "vmovaps %%ymm2, (%1)\n"
                        "vmovaps %%ymm3, 0x20(%1)\n"

                        : // No output operands
                        : "r" (networkn0 + i), "r" (networkg1 + (p * 128) + i), "x" (factor)
                        : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm15"
                    );
                }
            }
            //Hidden Layer 1 - Hidden Layer 0
            for (p = 0; p < 128; p++) {
                networkg0_neuron[p] = 0;
                for (q = 0; q < 32; q++) {
                    networkg0_neuron[p] += networkg1_neuron[q] * network1[(q * 128) + p];
                }
                if (networkb0[p] >= 0) {
                    networkg0_bi = networkg0_neuron[p];
                } else {
                    networkg0_neuron[p] *= exp(networkb0[p]);
                    networkg0_bi = networkg0_neuron[p];
                }
                _mm_prefetch((const char*)(networkg0 + (p * 784)), _MM_HINT_T0);
                factor = _mm256_set1_ps(networkg0_neuron[p]);
                for (i = 0; i <= 784 - 56; i += 56) {
                    __asm__ volatile (
                        // Load 7 ymm registers from a
                        "vmovaps (%0), %%ymm0\n"
                        "vmovaps 0x20(%0), %%ymm1\n"
                        "vmovaps 0x40(%0), %%ymm2\n"
                        "vmovaps 0x60(%0), %%ymm3\n"
                        "vmovaps 0x80(%0), %%ymm4\n"
                        "vmovaps 0xa0(%0), %%ymm5\n"
                        "vmovaps 0xc0(%0), %%ymm6\n"

                        // Load 7 ymm registers from b
                        "vmovaps (%1), %%ymm7\n"
                        "vmovaps 0x20(%1), %%ymm8\n"
                        "vmovaps 0x40(%1), %%ymm9\n"
                        "vmovaps 0x60(%1), %%ymm10\n"
                        "vmovaps 0x80(%1), %%ymm11\n"
                        "vmovaps 0xa0(%1), %%ymm12\n"
                        "vmovaps 0xc0(%1), %%ymm13\n"

                        // Load factor_vec into ymm14
                        "vmovaps %2, %%ymm14\n"

                        // Multiply a by factor and add to b using vfmadd231ps
                        "vfmadd231ps %%ymm14, %%ymm0, %%ymm7\n"
                        "vfmadd231ps %%ymm14, %%ymm1, %%ymm8\n"
                        "vfmadd231ps %%ymm14, %%ymm2, %%ymm9\n"
                        "vfmadd231ps %%ymm14, %%ymm3, %%ymm10\n"
                        "vfmadd231ps %%ymm14, %%ymm4, %%ymm11\n"
                        "vfmadd231ps %%ymm14, %%ymm5, %%ymm12\n"
                        "vfmadd231ps %%ymm14, %%ymm6, %%ymm13\n"

                        // Store results back to b
                        "vmovaps %%ymm7, (%1)\n"
                        "vmovaps %%ymm8, 0x20(%1)\n"
                        "vmovaps %%ymm9, 0x40(%1)\n"
                        "vmovaps %%ymm10, 0x60(%1)\n"
                        "vmovaps %%ymm11, 0x80(%1)\n"
                        "vmovaps %%ymm12, 0xa0(%1)\n"
                        "vmovaps %%ymm13, 0xc0(%1)\n"

                        : // No output operands
                        : "r" (train_image[thisDtId] + i), "r" (networkg0 + (p * 784) + i), "x" (factor)
                        : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14"
                    );
                }
            }
            //Clear Temp
            _mm_prefetch((const char*)(networkn0), _MM_HINT_T0);
            std::fill(networkn0, networkn0 + 128, 0);
            std::fill(networkn1, networkn1 + 32, 0);
            std::fill(networkn2, networkn2 + 10, 0);
            std::fill(networkb0, networkb0 + 128, 0);
            std::fill(networkb1, networkb1 + 32, 0);
            std::fill(networkb2, networkb2 + 10, 0);
        }

        //Mini-Batch Stochastic Gradient Descent (SGD)
        mtx_index_2 = 0;
        mtx_index_1 = 0;
        mtx_index_0 = 0;
        for (; !networkg2_todoList[0] || !networkg1_todoList[0] || !networkg0_todoList[0] || !networkg0_todoList[1] || !networkg0_todoList[2] || !networkg0_todoList[3] || !networkg0_todoList[4] || !networkg0_todoList[5] || !networkg0_todoList[6] || !networkg0_todoList[7];) {//Update All Blocks
            //Find a Free Block to update
            if (!networkg2_todoList[0]) {
                if (networkgs2_mtx[mtx_index_2].try_lock()) {
                    _mm_prefetch((const char*)(networkg2), _MM_HINT_T0);
                    for (p = 0; p < 10; p++) {
                        __asm__ volatile (
                            // Load 4 ymm registers from a
                            "vmovaps (%0), %%ymm0\n"
                            "vmovaps 0x20(%0), %%ymm1\n"
                            "vmovaps 0x40(%0), %%ymm2\n"
                            "vmovaps 0x60(%0), %%ymm3\n"

                            // Load 4 ymm registers from b
                            "vmovaps (%1), %%ymm4\n"
                            "vmovaps 0x20(%1), %%ymm5\n"
                            "vmovaps 0x40(%1), %%ymm6\n"
                            "vmovaps 0x60(%1), %%ymm7\n"

                            // Add corresponding ymm registers
                            "vaddps %%ymm4, %%ymm0, %%ymm0\n"
                            "vaddps %%ymm5, %%ymm1, %%ymm1\n"
                            "vaddps %%ymm6, %%ymm2, %%ymm2\n"
                            "vaddps %%ymm7, %%ymm3, %%ymm3\n"

                            // Store results back to memory
                            "vmovaps %%ymm0, (%1)\n"
                            "vmovaps %%ymm1, 0x20(%1)\n"
                            "vmovaps %%ymm2, 0x40(%1)\n"
                            "vmovaps %%ymm3, 0x60(%1)\n"

                            : // No output operands
                            : "r" (networkg2 + (p * 32)), "r" (networkgs2 + (p * 32))
                            : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7"
                        );
                    }
                    networkgs0_bi += networkg0_bi;
                    networkgs2_mtx[mtx_index_2].unlock();
                    networkg2_todoList[mtx_index_2] = true;
                }
            }
            if (!networkg1_todoList[0]) {
                if (networkgs1_mtx[mtx_index_1].try_lock()) {
                    _mm_prefetch((const char*)(networkg1), _MM_HINT_T0);
                    for (p = 0; p < 32; p++) {
                        for (i = 0; i <= 128 - 64; i += 64) {
                            __asm__ volatile (
                                // Load 8 ymm registers from a
                                "vmovaps (%0), %%ymm0\n"
                                "vmovaps 0x20(%0), %%ymm1\n"
                                "vmovaps 0x40(%0), %%ymm2\n"
                                "vmovaps 0x60(%0), %%ymm3\n"
                                "vmovaps 0x80(%0), %%ymm4\n"
                                "vmovaps 0xa0(%0), %%ymm5\n"
                                "vmovaps 0xc0(%0), %%ymm6\n"
                                "vmovaps 0xe0(%0), %%ymm7\n"

                                // Load 8 ymm registers from b
                                "vmovaps (%1), %%ymm8\n"
                                "vmovaps 0x20(%1), %%ymm9\n"
                                "vmovaps 0x40(%1), %%ymm10\n"
                                "vmovaps 0x60(%1), %%ymm11\n"
                                "vmovaps 0x80(%1), %%ymm12\n"
                                "vmovaps 0xa0(%1), %%ymm13\n"
                                "vmovaps 0xc0(%1), %%ymm14\n"
                                "vmovaps 0xe0(%1), %%ymm15\n"

                                // Add corresponding ymm registers
                                "vaddps %%ymm8, %%ymm0, %%ymm0\n"
                                "vaddps %%ymm9, %%ymm1, %%ymm1\n"
                                "vaddps %%ymm10, %%ymm2, %%ymm2\n"
                                "vaddps %%ymm11, %%ymm3, %%ymm3\n"
                                "vaddps %%ymm12, %%ymm4, %%ymm4\n"
                                "vaddps %%ymm13, %%ymm5, %%ymm5\n"
                                "vaddps %%ymm14, %%ymm6, %%ymm6\n"
                                "vaddps %%ymm15, %%ymm7, %%ymm7\n"

                                // Store results back to memory
                                "vmovaps %%ymm0, (%1)\n"
                                "vmovaps %%ymm1, 0x20(%1)\n"
                                "vmovaps %%ymm2, 0x40(%1)\n"
                                "vmovaps %%ymm3, 0x60(%1)\n"
                                "vmovaps %%ymm4, 0x80(%1)\n"
                                "vmovaps %%ymm5, 0xa0(%1)\n"
                                "vmovaps %%ymm6, 0xc0(%1)\n"
                                "vmovaps %%ymm7, 0xe0(%1)\n"

                                : // No output operands
                                : "r" (networkg1 + (p * 128) + i), "r" (networkgs1 + (p * 128) + i)
                                : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15"
                            );
                        }
                    }
                    networkgs1_bi += networkg1_bi;
                    networkgs1_mtx[mtx_index_1].unlock();
                    networkg1_todoList[mtx_index_1] = true;
                }
            }
            for (mtx_index_0 = 0; mtx_index_0 < 8; mtx_index_0++) {
                if (!networkg0_todoList[mtx_index_0]) {
                    if (networkgs0_mtx[mtx_index_0].try_lock()) {
                        if(mtx_index_0 == 7){
                          networkgs0_bi += networkg0_bi;
                        }
                        _mm_prefetch((const char*)(networkg0), _MM_HINT_T0);
                        for (p = 0; p < 16; p++) {
                            for (i = 0; i <= 784 - 64; i += 64) {
                                __asm__ volatile (
                                    // Load 8 ymm registers from a
                                    "vmovaps (%0), %%ymm0\n"
                                    "vmovaps 0x20(%0), %%ymm1\n"
                                    "vmovaps 0x40(%0), %%ymm2\n"
                                    "vmovaps 0x60(%0), %%ymm3\n"
                                    "vmovaps 0x80(%0), %%ymm4\n"
                                    "vmovaps 0xa0(%0), %%ymm5\n"
                                    "vmovaps 0xc0(%0), %%ymm6\n"
                                    "vmovaps 0xe0(%0), %%ymm7\n"

                                    // Load 8 ymm registers from b
                                    "vmovaps (%1), %%ymm8\n"
                                    "vmovaps 0x20(%1), %%ymm9\n"
                                    "vmovaps 0x40(%1), %%ymm10\n"
                                    "vmovaps 0x60(%1), %%ymm11\n"
                                    "vmovaps 0x80(%1), %%ymm12\n"
                                    "vmovaps 0xa0(%1), %%ymm13\n"
                                    "vmovaps 0xc0(%1), %%ymm14\n"
                                    "vmovaps 0xe0(%1), %%ymm15\n"

                                    // Add corresponding ymm registers
                                    "vaddps %%ymm8, %%ymm0, %%ymm0\n"
                                    "vaddps %%ymm9, %%ymm1, %%ymm1\n"
                                    "vaddps %%ymm10, %%ymm2, %%ymm2\n"
                                    "vaddps %%ymm11, %%ymm3, %%ymm3\n"
                                    "vaddps %%ymm12, %%ymm4, %%ymm4\n"
                                    "vaddps %%ymm13, %%ymm5, %%ymm5\n"
                                    "vaddps %%ymm14, %%ymm6, %%ymm6\n"
                                    "vaddps %%ymm15, %%ymm7, %%ymm7\n"

                                    // Store results back to memory
                                    "vmovaps %%ymm0, (%1)\n"
                                    "vmovaps %%ymm1, 0x20(%1)\n"
                                    "vmovaps %%ymm2, 0x40(%1)\n"
                                    "vmovaps %%ymm3, 0x60(%1)\n"
                                    "vmovaps %%ymm4, 0x80(%1)\n"
                                    "vmovaps %%ymm5, 0xa0(%1)\n"
                                    "vmovaps %%ymm6, 0xc0(%1)\n"
                                    "vmovaps %%ymm7, 0xe0(%1)\n"

                                    : // No output operands
                                    : "r" (networkg0 + ((mtx_index_0 * 16 + p) * 784) + i), "r" (networkgs0 + ((mtx_index_0 * 16 + p) * 784) + i)
                                    : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15"
                                );
                            }
                            __asm__ volatile (
                                // Load 2 ymm registers from a
                                "vmovaps (%0), %%ymm0\n"
                                "vmovaps 0x20(%0), %%ymm1\n"

                                // Load 2 ymm registers from b
                                "vmovaps (%1), %%ymm2\n"
                                "vmovaps 0x20(%1), %%ymm3\n"

                                // Add corresponding ymm registers
                                "vaddps %%ymm2, %%ymm0, %%ymm0\n"
                                "vaddps %%ymm3, %%ymm1, %%ymm1\n"

                                // Store results back to memory
                                "vmovaps %%ymm0, (%1)\n"
                                "vmovaps %%ymm1, 0x20(%1)\n"

                                : // No output operands
                                : "r" (networkg0 + ((mtx_index_0 * 16 + p) * 784) + i), "r" (networkgs0 + ((mtx_index_0 * 16 + p) * 784) + i)
                                : "memory", "ymm0", "ymm1", "ymm2", "ymm3"
                            );
                        }
                        networkgs0_bi += networkg0_bi;
                        networkgs0_mtx[mtx_index_0].unlock();
                        networkg0_todoList[mtx_index_0] = true;
                    }
                }
            }
        }
        //Init todoList
        networkg2_todoList[0] = false;
        networkg1_todoList[0] = false;
        for (i = 0; i < 8; i++) {
            networkg0_todoList[i] = false;
        }

        //Clear Temp
        _mm_prefetch((const char*)(networkg0), _MM_HINT_T0);
        std::fill(networkg0, networkg0 + 100352, 0);
        _mm_prefetch((const char*)(networkg1), _MM_HINT_T0);
        std::fill(networkg1, networkg1 + 4096, 0);
        _mm_prefetch((const char*)(networkg2), _MM_HINT_T0);
        std::fill(networkg2, networkg2 + 320, 0);
        networkg0_bi = 0.0f;
        networkg1_bi = 0.0f;
        networkg2_bi = 0.0f;

        mtx.lock();
        MSETotal += MSError;//Add Lost to Global Cost of this Batch
        reportI++;//This Thread Finished BP
        if (reportI == tpool->TSize) {//If All Threads Finished BP
            //Update Weights & Bias
            _mm_prefetch((const char*)(networkg0), _MM_HINT_T0);
            memcpy(network0, networkgs0, 401408);
            _mm_prefetch((const char*)(networkg1), _MM_HINT_T0);
            memcpy(network1, networkgs1, 16384);
            _mm_prefetch((const char*)(networkg2), _MM_HINT_T0);
            memcpy(network2, networkgs2, 1280);
            network0_bi = networkgs0_bi;
            network1_bi = networkgs1_bi;
            network2_bi = networkgs2_bi;
            err += MSETotal;//Add Batch Cost to Epoch Cost
            MSETotal = 0;//Clear temp
            reportI = 0;//Clear temp
            gate = true;//Ready for next Batch
        }
        mtx.unlock();
        MSError = 0;
        tpool->TFree[TId].store(true);//This Thread is Free
    }
}


inline std::vector<std::thread> TP::init(int size) {
    TSize = size;
    std::vector<std::thread> threads;
    for (int i = 0; i < size; i++) {
        TGate.emplace_back();
        TFree.emplace_back(true);
        threads.push_back(thread(trainNet, i));
    }
    return threads;
}


void train(float rate, float aim) {
    std::cout << "Gradient loss function: Cross Entropy" << std::endl << "------------------------------" << std::endl;
    int i{}, c, w;
    std::chrono::duration<double> duration;
    vector<vector<float>> temp0(batchSize / tpool->TSize);
    vector<vector<float>> temp1(batchSize / tpool->TSize);
    while (true) {
        i++;
        err = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (c = 0; c < dts / batchSize; c++) {
            while (true) {
                if (gate == true) {
                    gate = false;
                    BId = c;
                    for (w = 0; w < batchSize; w += batchSize / tpool->TSize) {
                        tpool->add();
                    }
                    break;
                }
            }
        }
        while (gate == false);
        auto end = std::chrono::high_resolution_clock::now();
        duration = end - start;

        rate *= 1.1f;

        if (err <= aim) {
            std::cout << "------------------------------" << std::endl;
            std::cout << ">>> finished " << dts * i << " steps (" << i << " Epoch) gradient descent (Cost: " << err << ")" << std::endl;
            break;
        } else {
            std::cout << "Epoch " << i << " | Time " << duration.count() << " | MSE " << err << "  " << err / dts << " | rate " << rate << std::endl;
        }
    }
}



int main() {
    std::cout << "1/3 Init Model" << endl;
    gate = true;
    MSETotal = 0;
    

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.2, 0.2);
    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 784; ++j) {
            network0[i*784 + j] = dis(gen);
        }
    }
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 128; ++j) {
            network1[i*128+j] = dis(gen);
        }
    }
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 32; ++j) {
            network2[i*32+j] = dis(gen);
        }
    }
    memcpy(networkgs0, network0, 401408);
    memcpy(networkgs1, network1, 16384);
    memcpy(networkgs2, network2, 1280);

    batchSize = 50;
    std::vector<std::thread> threads = tpool->init(10);//BatchSize must be a positive integer multiple of the number of threads

    rate = 0.003;//Learning Rate
    aim = 1;//Aiming Loss(MSE in total)
    dts = 50000;//Data Limited(Max 60000)

    std::cout << "2/3 Load Training Data" << endl;

    char train_image_name[] = "train-images.idx3-ubyte";
    char train_label_name[] = "train-labels.idx1-ubyte";
    vector< vector<int> > train_feature_vector;
    read_Mnist_Images(train_image_name, train_feature_vector, dts);
    convert_array_image(train_feature_vector);
    vector<int> train_labels;
    read_Mnist_Label(train_label_name, train_labels, dts);
    std::cout << "3/3 Convert Training Data" << endl;
    convert_array_label(train_labels);

    std::cout << "Ready..." << std::endl;
    train(rate, aim);

    for (int i = 0; i < threads.size(); i++) {
        threads[i].detach();
    }
}