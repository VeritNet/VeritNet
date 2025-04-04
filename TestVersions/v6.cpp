/*
* The codes are generated with Blueprint By VeritNet Engine, using AVX2, so I manually added comments and modified some variable names to make the codes easier to read.
* g++ -O3 -march=native -funroll-all-loops -mavx2 -o v6.exe v6.cpp
* Version 2024.10.7.6
* [128 Elu, 32 Elu, 10 Softmax]
* 注意若要修改线程数量需重分块
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
#include <immintrin.h>
#include <unistd.h>
#include "MNIST.h"


using namespace std;


std::mutex mtx;//Global Mutex Lock
//Thread Pool
class TP {
public:
    vector<mutex*> TGate;//Wake Conditon
    deque<atomic<bool>> TFree;//Is Free
    int TSize;//Num of threads
    std::vector<std::thread> init(int size);//Create & Detach
    inline void add() {//Add Task
        bool noBreak = true;
        while (noBreak) {
            for (int i = 0; i < TSize; i++) {//Find a Free Thread
                if (TFree[i].load()) {
                    TFree[i].store(false);
                    TGate[i]->unlock();
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
float* network0 = static_cast<float*>(_mm_malloc(128*784 * sizeof(float), 64));
float* network1 = static_cast<float*>(_mm_malloc(32*128 * sizeof(float), 64));
float* network2 = static_cast<float*>(_mm_malloc(10*32 * sizeof(float), 64));
float* network0_bi = static_cast<float*>(_mm_malloc(128 * sizeof(float), 64));
float* network1_bi = static_cast<float*>(_mm_malloc(32 * sizeof(float), 64));
float* network2_bi = static_cast<float*>(_mm_malloc(10 * sizeof(float), 64));

float MSETotal;//MSE Cost

//Network Gradients (Shared Memory)
float* networkgs0 = static_cast<float*>(_mm_malloc(128*784 * sizeof(float), 64));
float* networkgs1 = static_cast<float*>(_mm_malloc(32*128 * sizeof(float), 64));
float* networkgs2 = static_cast<float*>(_mm_malloc(10*32 * sizeof(float), 64));
float* networkgs0_bi = static_cast<float*>(_mm_malloc(128 * sizeof(float), 64));
float* networkgs1_bi = static_cast<float*>(_mm_malloc(32 * sizeof(float), 64));
float* networkgs2_bi = static_cast<float*>(_mm_malloc(10 * sizeof(float), 64));

//Locks for Blocks
std::vector<std::mutex> networkgs0_mtx(16);
std::vector<std::mutex> networkgs1_mtx(1);
std::vector<std::mutex> networkgs2_mtx(1);
bool gate;//Main Thread Gate
int reportI = 0;
int BId = 0;//Batch Id
float rate, aim, err;//Learning Rate, MSE aim, Cost of 1 Epoch


#define forward_a(i) \
sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i), _mm256_load_ps(train_image[thisDtId] + i), sum);\
sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 8), _mm256_load_ps(train_image[thisDtId] + i + 8), sum);\
sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 16), _mm256_load_ps(train_image[thisDtId] + i + 16), sum);\
sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 24), _mm256_load_ps(train_image[thisDtId] + i + 24), sum);\
sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 32), _mm256_load_ps(train_image[thisDtId] + i + 32), sum);\
sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 40), _mm256_load_ps(train_image[thisDtId] + i + 40), sum);\
sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 48), _mm256_load_ps(train_image[thisDtId] + i + 48), sum);\
sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 56), _mm256_load_ps(train_image[thisDtId] + i + 56), sum);\

#define bp_a(i) \
__asm__ volatile (\
    "vmovaps (%0), %%ymm0\n"\
    "vmovaps 0x20(%0), %%ymm1\n"\
    "vmovaps 0x40(%0), %%ymm2\n"\
    "vmovaps 0x60(%0), %%ymm3\n"\
    "vmovaps 0x80(%0), %%ymm4\n"\
    "vmovaps 0xa0(%0), %%ymm5\n"\
    "vmovaps 0xc0(%0), %%ymm6\n"\
    "vmovaps (%1), %%ymm7\n"\
    "vmovaps 0x20(%1), %%ymm8\n"\
    "vmovaps 0x40(%1), %%ymm9\n"\
    "vmovaps 0x60(%1), %%ymm10\n"\
    "vmovaps 0x80(%1), %%ymm11\n"\
    "vmovaps 0xa0(%1), %%ymm12\n"\
    "vmovaps 0xc0(%1), %%ymm13\n"\
    "vmovaps %2, %%ymm14\n"\
    "vfmadd231ps %%ymm14, %%ymm0, %%ymm7\n"\
    "vfmadd231ps %%ymm14, %%ymm1, %%ymm8\n"\
    "vfmadd231ps %%ymm14, %%ymm2, %%ymm9\n"\
    "vfmadd231ps %%ymm14, %%ymm3, %%ymm10\n"\
    "vfmadd231ps %%ymm14, %%ymm4, %%ymm11\n"\
    "vfmadd231ps %%ymm14, %%ymm5, %%ymm12\n"\
    "vfmadd231ps %%ymm14, %%ymm6, %%ymm13\n"\
    "vmovaps %%ymm7, (%1)\n"\
    "vmovaps %%ymm8, 0x20(%1)\n"\
    "vmovaps %%ymm9, 0x40(%1)\n"\
    "vmovaps %%ymm10, 0x60(%1)\n"\
    "vmovaps %%ymm11, 0x80(%1)\n"\
    "vmovaps %%ymm12, 0xa0(%1)\n"\
    "vmovaps %%ymm13, 0xc0(%1)\n"\
    : \
    : "r" (train_image[thisDtId] + i), "r" (networkg0 + (p * 784) + i), "x" (factor)\
    : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14"\
);

#define up_a(i) \
__asm__ volatile (\
    "vmovaps (%0), %%ymm0\n"\
    "vmovaps 0x20(%0), %%ymm1\n"\
    "vmovaps 0x40(%0), %%ymm2\n"\
    "vmovaps 0x60(%0), %%ymm3\n"\
    "vmovaps 0x80(%0), %%ymm4\n"\
    "vmovaps 0xa0(%0), %%ymm5\n"\
    "vmovaps 0xc0(%0), %%ymm6\n"\
    "vmovaps 0xe0(%0), %%ymm7\n"\
    "vmovaps (%1), %%ymm8\n"\
    "vmovaps 0x20(%1), %%ymm9\n"\
    "vmovaps 0x40(%1), %%ymm10\n"\
    "vmovaps 0x60(%1), %%ymm11\n"\
    "vmovaps 0x80(%1), %%ymm12\n"\
    "vmovaps 0xa0(%1), %%ymm13\n"\
    "vmovaps 0xc0(%1), %%ymm14\n"\
    "vmovaps 0xe0(%1), %%ymm15\n"\
    "vaddps %%ymm8, %%ymm0, %%ymm0\n"\
    "vaddps %%ymm9, %%ymm1, %%ymm1\n"\
    "vaddps %%ymm10, %%ymm2, %%ymm2\n"\
    "vaddps %%ymm11, %%ymm3, %%ymm3\n"\
    "vaddps %%ymm12, %%ymm4, %%ymm4\n"\
    "vaddps %%ymm13, %%ymm5, %%ymm5\n"\
    "vaddps %%ymm14, %%ymm6, %%ymm6\n"\
    "vaddps %%ymm15, %%ymm7, %%ymm7\n"\
    "vmovaps %%ymm0, (%1)\n"\
    "vmovaps %%ymm1, 0x20(%1)\n"\
    "vmovaps %%ymm2, 0x40(%1)\n"\
    "vmovaps %%ymm3, 0x60(%1)\n"\
    "vmovaps %%ymm4, 0x80(%1)\n"\
    "vmovaps %%ymm5, 0xa0(%1)\n"\
    "vmovaps %%ymm6, 0xc0(%1)\n"\
    "vmovaps %%ymm7, 0xe0(%1)\n"\
    : \
    : "r" (networkg0 + ((mtx_index_0 * 8 + p) * 784) + i), "r" (networkgs0 + ((mtx_index_0 * 8 + p) * 784) + i)\
    : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15"\
);



inline void trainNet(int TId/*Thread Id*/) {
    tpool->TGate[TId] = new mutex;
    tpool->TGate[TId]->lock();
    int thisDtId;

    //Block update of gradients in shared memory
    vector<bool> networkg0_todoList(16);
    vector<bool> networkg1_todoList(1);
    vector<bool> networkg2_todoList(1);
    
    float MSError{};//Error Sum of All Data in this Epoch in this Thread

    float SSum;//For Softmax Activation
    int dtIndex;//Loop Index
    int mtx_index_0, mtx_index_1, mtx_index_2;//Lock Index
    __m256 sum, factor, mask,
        exp_temp, exp_fx, exp_floor_fx, exp_x_squared;//For SIMD
    __m128 sum_high, sum_low;//For SIMD

    const __m256 EXP_HI = _mm256_set1_ps(88.3762626647949f);
    const __m256 EXP_LO = _mm256_set1_ps(-88.3762626647949f);
    const __m256 LOG2EF = _mm256_set1_ps(1.44269504088896341f);
    const __m256 EXP_C1 = _mm256_set1_ps(-0.693359375f);
    const __m256 EXP_C2 = _mm256_set1_ps(2.12194440e-4f);
    const __m256 EXP_POLY_COEFF_0 = _mm256_set1_ps(1.9875691500E-4f);
    const __m256 EXP_POLY_COEFF_1 = _mm256_set1_ps(1.3981999507E-3f);
    const __m256 EXP_POLY_COEFF_2 = _mm256_set1_ps(8.3334519073E-3f);
    const __m256 EXP_POLY_COEFF_3 = _mm256_set1_ps(4.1665795894E-2f);
    const __m256 EXP_POLY_COEFF_4 = _mm256_set1_ps(1.6666665459E-1f);
    const __m256 EXP_POLY_COEFF_5 = _mm256_set1_ps(5.0000001201E-1f);
    const __m256 padding1 = _mm256_set1_ps(1);
    const __m256 padding_half = _mm256_set1_ps(0.5f);
    const __m256i INT_7F = _mm256_set1_epi32(0x7F);

    float* networkg0 = static_cast<float*>(_mm_malloc(128*784 * sizeof(float), 64));//Network Gradients (Thread Memory)
    float* networkg0_bi = static_cast<float*>(_mm_malloc(128 * sizeof(float), 64));
    float* networkb0 = static_cast<float*>(_mm_malloc(128 * sizeof(float), 64));//w*x + b (Before Activation) for hidden layer 0
    float* networkn0 = static_cast<float*>(_mm_malloc(128 * sizeof(float), 64));//Activation for hidden layer 0

    float* networkg1 = static_cast<float*>(_mm_malloc(32*128 * sizeof(float), 64));//Network Gradients (Thread Memory)
    float* networkg1_bi = static_cast<float*>(_mm_malloc(32 * sizeof(float), 64));
    float* networkb1 = static_cast<float*>(_mm_malloc(32 * sizeof(float), 64));//..for hidden layer 1
    float* networkn1 = static_cast<float*>(_mm_malloc(32 * sizeof(float), 64));//..for hidden layer 1

    float* networkg2 = static_cast<float*>(_mm_malloc(10*32 * sizeof(float), 64));//Network Gradients (Thread Memory)
    float* networkg2_bi = static_cast<float*>(_mm_malloc(10 * sizeof(float), 64));
    float* networkb2 = static_cast<float*>(_mm_malloc(10 * sizeof(float), 64));//..for hidden layer 2
    float* networkn2 = static_cast<float*>(_mm_malloc(10 * sizeof(float), 64));//..for hidden layer 2

    std::fill(networkg0, networkg0 + 100352, 0);
    std::fill(networkg0_bi, networkg0_bi + 128, 0);
    std::fill(networkg1, networkg1 + 4096, 0);
    std::fill(networkg1_bi, networkg1_bi + 32, 0);
    std::fill(networkg2, networkg2 + 320, 0);
    std::fill(networkg2_bi, networkg2_bi + 10, 0);

    std::fill(networkn0, networkn0 + 128, 0);
    std::fill(networkn1, networkn1 + 32, 0);
    std::fill(networkn2, networkn2 + 10, 0);
    std::fill(networkb0, networkb0 + 128, 0);
    std::fill(networkb1, networkb1 + 32, 0);
    std::fill(networkb2, networkb2 + 10, 0);

    //The gradient of neurons is equal to the gradient of neuron bias
    float* networkg0_neuron = static_cast<float*>(_mm_malloc(128 * sizeof(float), 64));
    float* networkg1_neuron = static_cast<float*>(_mm_malloc(32 * sizeof(float), 64));
    float* networkg2_neuron = static_cast<float*>(_mm_malloc(10 * sizeof(float), 64));
    std::fill(networkg0_neuron, networkg0_neuron + 128, 0);
    std::fill(networkg1_neuron, networkg1_neuron + 32, 0);
    std::fill(networkg2_neuron, networkg2_neuron + 10, 0);


    for (;;) {
        tpool->TGate[TId]->lock();
        for (dtIndex = batchSize / tpool->TSize - 1; dtIndex >= 0; dtIndex--) {//Train all data in this task
            thisDtId = (batchSize * BId) + (TId * (batchSize / tpool->TSize)) + dtIndex;
            //Feed Forward
            //Input Layer - Hidden Layer 0
            for (int p = 0; p < 128; p++) {
                sum = _mm256_setzero_ps();
                //int i = 0;
                //_mm_prefetch((const char*)(network0 + (p * 784)), _MM_HINT_T0);
                forward_a(0)
                forward_a(64)
                forward_a(128)
                forward_a(192)
                forward_a(256)
                forward_a(320)
                forward_a(384)
                forward_a(448)
                forward_a(512)
                forward_a(576)
                forward_a(640)
                forward_a(704)
                //_mm_prefetch((const char*)(network0 + i + 16), _MM_HINT_T0);
                sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + 768), _mm256_load_ps(train_image[thisDtId] + 768), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + 768 + 8), _mm256_load_ps(train_image[thisDtId] + 768 + 8), sum);

                sum_high = _mm256_extractf128_ps(sum, 1);
                sum_low = _mm256_extractf128_ps(sum, 0);
                sum_low = _mm_add_ps(sum_low, sum_high);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                networkb0[p] = _mm_cvtss_f32(sum_low);
            }
            for (int p = 0; p <= 128 - 64; p += 64) {
                _mm256_store_ps(networkb0 + p, _mm256_add_ps(_mm256_load_ps(networkb0 + p), _mm256_load_ps(network0_bi + p)));
                _mm256_store_ps(networkb0 + p + 8, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 8), _mm256_load_ps(network0_bi + p + 8)));
                _mm256_store_ps(networkb0 + p + 16, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 16), _mm256_load_ps(network0_bi + p + 16)));
                _mm256_store_ps(networkb0 + p + 24, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 24), _mm256_load_ps(network0_bi + p + 24)));
                _mm256_store_ps(networkb0 + p + 32, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 32), _mm256_load_ps(network0_bi + p + 32)));
                _mm256_store_ps(networkb0 + p + 40, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 40), _mm256_load_ps(network0_bi + p + 40)));
                _mm256_store_ps(networkb0 + p + 48, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 48), _mm256_load_ps(network0_bi + p + 48)));
                _mm256_store_ps(networkb0 + p + 56, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 56), _mm256_load_ps(network0_bi + p + 56)));
            }
            for (int p = 0; p < 128; p++) {
                if (networkb0[p] >= 0) {
                    networkn0[p] = networkb0[p];
                } else {
                    networkn0[p] = exp(networkb0[p]) - 1;
                }
            }

            //Hidden Layer 0 - Hidden Layer 1
            for (int p = 0; p < 32; p++) {
                sum = _mm256_setzero_ps();
                int i = 0;
                //_mm_prefetch((const char*)(network1), _MM_HINT_T0);
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
            }
            _mm256_store_ps(networkb1, _mm256_add_ps(_mm256_load_ps(networkb1), _mm256_load_ps(network1_bi)));
            _mm256_store_ps(networkb1 + 8, _mm256_add_ps(_mm256_load_ps(networkb1 + 8), _mm256_load_ps(network1_bi + 8)));
            _mm256_store_ps(networkb1 + 16, _mm256_add_ps(_mm256_load_ps(networkb1 + 16), _mm256_load_ps(network1_bi + 16)));
            _mm256_store_ps(networkb1 + 24, _mm256_add_ps(_mm256_load_ps(networkb1 + 24), _mm256_load_ps(network1_bi + 24)));
            for (int p = 0; p < 32; p++) {
                if (networkb1[p] >= 0) {
                    networkn1[p] = networkb1[p];
                } else {
                    networkn1[p] = exp(networkb1[p]) - 1;
                }
            }

            //Hidden Layer 1 - Hidden Layer 2
            for (int p = 0; p < 10; p++) {
                sum = _mm256_setzero_ps();
                int i = 0;
                //_mm_prefetch((const char*)(network2), _MM_HINT_T0);
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
            }
            _mm256_store_ps(networkb2, _mm256_add_ps(_mm256_load_ps(networkb2), _mm256_load_ps(network2_bi)));
            networkb2[8] += network2_bi[8];
            networkb2[9] += network2_bi[9];

            //Hidden Layer 2 - Output Layer
            /*SSum = 0;
            for (i = 0; i < 10; i++) {
                SSum += exp(networkb2[i]);
            }
            for (i = 0; i < 10; i++) {
                networkn2[i] += exp(networkb2[i]) / SSum;
            }*/
            
            SSum = 0;
            exp_temp = _mm256_min_ps(_mm256_load_ps(networkb2), EXP_HI);
            exp_temp = _mm256_max_ps(exp_temp, EXP_LO);
            exp_fx = _mm256_fmadd_ps(exp_temp, LOG2EF, padding_half);
            exp_floor_fx = _mm256_floor_ps(exp_fx);
            mask = _mm256_cmp_ps(exp_floor_fx, exp_fx, _CMP_GT_OS);
            mask = _mm256_and_ps(mask, padding1);
            exp_fx = _mm256_sub_ps(exp_floor_fx, mask);
            exp_temp = _mm256_fmadd_ps(exp_fx, EXP_C2, exp_temp);
            exp_temp = _mm256_fmadd_ps(exp_fx, EXP_C1, exp_temp);
            exp_x_squared = _mm256_mul_ps(exp_temp, exp_temp);
            exp_temp = _mm256_fmadd_ps(EXP_POLY_COEFF_5, exp_x_squared, exp_temp);
            exp_temp = _mm256_fmadd_ps(EXP_POLY_COEFF_4, exp_x_squared, exp_temp);
            exp_temp = _mm256_fmadd_ps(EXP_POLY_COEFF_3, exp_x_squared, exp_temp);
            exp_temp = _mm256_fmadd_ps(EXP_POLY_COEFF_2, exp_x_squared, exp_temp);
            exp_temp = _mm256_fmadd_ps(EXP_POLY_COEFF_1, exp_x_squared, exp_temp);
            exp_temp = _mm256_fmadd_ps(EXP_POLY_COEFF_0, exp_x_squared, exp_temp);
            exp_temp = _mm256_add_ps(exp_temp, padding1);
            exp_temp = _mm256_mul_ps(exp_temp, _mm256_castsi256_ps(
                _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(exp_fx), INT_7F), 23)
            ));
            sum_high = _mm256_extractf128_ps(exp_temp, 1);
            sum_low = _mm256_extractf128_ps(exp_temp, 0);
            sum_low = _mm_add_ps(sum_low, sum_high);
            sum_low = _mm_hadd_ps(sum_low, sum_low);
            sum_low = _mm_hadd_ps(sum_low, sum_low);
            SSum = _mm_cvtss_f32(sum_low) + exp(networkb2[8]) + exp(networkb2[9]);
            factor = _mm256_set1_ps(SSum);
            _mm256_store_ps(networkn2, _mm256_div_ps(exp_temp, factor));
            networkn2[8] = exp(networkb2[8]) / SSum;
            networkn2[9] = exp(networkb2[9]) / SSum;
            
            
            //Loss
            for (int p = 0; p < 10; p++) {
                MSError += ((train_label[thisDtId][p] - networkn2[p]) * (train_label[thisDtId][p] - networkn2[p]));
            }

            //Back Propagation
            //Output Layer - Hidden Layer 2
            for (int p = 0; p < 10; p++) {
                networkg2_neuron[p] = -rate * (networkn2[p] - train_label[thisDtId][p]);
                //_mm_prefetch((const char*)(networkg2 + (p * 32)), _MM_HINT_T0);
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
            __asm__ volatile (
                "vmovaps (%0), %%ymm0\n"
                "vmovaps (%1), %%ymm1\n"
                "vaddps %%ymm1, %%ymm0, %%ymm0\n"
                "vmovaps %%ymm0, (%1)\n"
                : // No output operands
                : "r" (networkg2_neuron), "r" (networkg2_bi)
                : "memory", "ymm0", "ymm1"
            );
            networkg2_bi[8] += networkg2_neuron[8];
            networkg2_bi[9] += networkg2_neuron[9];

            //Hidden Layer 2 - Hidden Layer 1
            for (int p = 0; p < 32; p++) {
                networkg1_neuron[p] = 0;
                for (int q = 0; q < 10; q++) {
                    networkg1_neuron[p] += networkg2_neuron[q] * network2[(q * 32) + p];
                }
                if (networkb1[p] < 0) {
                    networkg1_neuron[p] *= exp(networkb1[p]);
                }
                //_mm_prefetch((const char*)(networkg1 + (p * 128)), _MM_HINT_T0);
                factor = _mm256_set1_ps(networkg1_neuron[p]);
                int i = 0;
                for (; i <= 128 - 56; i += 56) {
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
                }
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
            __asm__ volatile (
                "vmovaps (%0), %%ymm0\n"
                "vmovaps 0x20(%0), %%ymm1\n"
                "vmovaps 0x40(%0), %%ymm2\n"
                "vmovaps 0x60(%0), %%ymm3\n"

                "vmovaps (%1), %%ymm4\n"
                "vmovaps 0x20(%1), %%ymm5\n"
                "vmovaps 0x40(%1), %%ymm6\n"
                "vmovaps 0x60(%1), %%ymm7\n"

                "vaddps %%ymm4, %%ymm0, %%ymm0\n"
                "vaddps %%ymm5, %%ymm1, %%ymm1\n"
                "vaddps %%ymm6, %%ymm2, %%ymm2\n"
                "vaddps %%ymm7, %%ymm3, %%ymm3\n"

                "vmovaps %%ymm0, (%1)\n"
                "vmovaps %%ymm1, 0x20(%1)\n"
                "vmovaps %%ymm2, 0x40(%1)\n"
                "vmovaps %%ymm3, 0x60(%1)\n"

                : // No output operands
                : "r" (networkg1_neuron), "r" (networkg1_bi)
                : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7"
            );

            //Hidden Layer 1 - Hidden Layer 0
            for (int p = 0; p < 128; p++) {
                networkg0_neuron[p] = 0;
                for (int q = 0; q < 32; q++) {
                    networkg0_neuron[p] += networkg1_neuron[q] * network1[(q * 128) + p];
                }
                if (networkb0[p] < 0) {
                    networkg0_neuron[p] *= exp(networkb0[p]);
                }
                //_mm_prefetch((const char*)(networkg0 + (p * 784)), _MM_HINT_T0);
                factor = _mm256_set1_ps(networkg0_neuron[p]);
                bp_a(0)
                bp_a(56)
                bp_a(112)
                bp_a(168)
                bp_a(224)
                bp_a(280)
                bp_a(336)
                bp_a(392)
                bp_a(448)
                bp_a(504)
                bp_a(560)
                bp_a(616)
                bp_a(672)
                bp_a(728)
            }
            for (int p = 0; p < 128; p += 64) {
                __asm__ volatile (
                    "vmovaps (%0), %%ymm0\n"
                    "vmovaps 0x20(%0), %%ymm1\n"
                    "vmovaps 0x40(%0), %%ymm2\n"
                    "vmovaps 0x60(%0), %%ymm3\n"
                    "vmovaps 0x80(%0), %%ymm4\n"
                    "vmovaps 0xa0(%0), %%ymm5\n"
                    "vmovaps 0xc0(%0), %%ymm6\n"
                    "vmovaps 0xe0(%0), %%ymm7\n"

                    "vmovaps (%1), %%ymm8\n"
                    "vmovaps 0x20(%1), %%ymm9\n"
                    "vmovaps 0x40(%1), %%ymm10\n"
                    "vmovaps 0x60(%1), %%ymm11\n"
                    "vmovaps 0x80(%1), %%ymm12\n"
                    "vmovaps 0xa0(%1), %%ymm13\n"
                    "vmovaps 0xc0(%1), %%ymm14\n"
                    "vmovaps 0xe0(%1), %%ymm15\n"

                    "vaddps %%ymm8, %%ymm0, %%ymm0\n"
                    "vaddps %%ymm9, %%ymm1, %%ymm1\n"
                    "vaddps %%ymm10, %%ymm2, %%ymm2\n"
                    "vaddps %%ymm11, %%ymm3, %%ymm3\n"
                    "vaddps %%ymm12, %%ymm4, %%ymm4\n"
                    "vaddps %%ymm13, %%ymm5, %%ymm5\n"
                    "vaddps %%ymm14, %%ymm6, %%ymm6\n"
                    "vaddps %%ymm15, %%ymm7, %%ymm7\n"

                    "vmovaps %%ymm0, (%1)\n"
                    "vmovaps %%ymm1, 0x20(%1)\n"
                    "vmovaps %%ymm2, 0x40(%1)\n"
                    "vmovaps %%ymm3, 0x60(%1)\n"
                    "vmovaps %%ymm4, 0x80(%1)\n"
                    "vmovaps %%ymm5, 0xa0(%1)\n"
                    "vmovaps %%ymm6, 0xc0(%1)\n"
                    "vmovaps %%ymm7, 0xe0(%1)\n"

                    : // No output operands
                    : "r" (networkg0_neuron + p), "r" (networkg0_bi + p)
                    : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14", "ymm15"
                );
            }


            //Clear Temp
            //_mm_prefetch((const char*)(networkb0), _MM_HINT_T0);
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
        for (; !networkg2_todoList[0] || !networkg1_todoList[0] || !networkg0_todoList[0] || !networkg0_todoList[1] || !networkg0_todoList[2] || !networkg0_todoList[3] || !networkg0_todoList[4] || !networkg0_todoList[5] || !networkg0_todoList[6] || !networkg0_todoList[7]
                 || !networkg0_todoList[8] || !networkg0_todoList[9] || !networkg0_todoList[10] || !networkg0_todoList[11] || !networkg0_todoList[12] || !networkg0_todoList[13] || !networkg0_todoList[14] || !networkg0_todoList[15];) {//Update All Blocks
            //Find a Free Block to update
            if (!networkg2_todoList[0]) {
                if (networkgs2_mtx[mtx_index_2].try_lock()) {
                    //_mm_prefetch((const char*)(networkg2), _MM_HINT_T0);
                    for (int p = 0; p < 10; p++) {
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
                    _mm256_store_ps(networkgs2_bi, _mm256_add_ps(_mm256_load_ps(networkgs2_bi), _mm256_load_ps(networkg2_bi)));
                    networkgs2_bi[8] += networkg2_bi[8];
                    networkgs2_bi[9] += networkg2_bi[9];
                    networkgs2_mtx[mtx_index_2].unlock();
                    networkg2_todoList[mtx_index_2] = true;
                }
            }
            if (!networkg1_todoList[0]) {
                if (networkgs1_mtx[mtx_index_1].try_lock()) {
                    //_mm_prefetch((const char*)(networkg1), _MM_HINT_T0);
                    for (int p = 0; p < 32; p++) {
                        for (int i = 0; i <= 128 - 64; i += 64) {
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
                    _mm256_store_ps(networkgs1_bi, _mm256_add_ps(_mm256_load_ps(networkgs1_bi), _mm256_load_ps(networkg1_bi)));
                    _mm256_store_ps(networkgs1_bi + 8, _mm256_add_ps(_mm256_load_ps(networkgs1_bi + 8), _mm256_load_ps(networkg1_bi + 8)));
                    _mm256_store_ps(networkgs1_bi + 16, _mm256_add_ps(_mm256_load_ps(networkgs1_bi + 16), _mm256_load_ps(networkg1_bi + 16)));
                    _mm256_store_ps(networkgs1_bi + 24, _mm256_add_ps(_mm256_load_ps(networkgs1_bi + 24), _mm256_load_ps(networkg1_bi + 24)));
                    networkgs1_mtx[mtx_index_1].unlock();
                    networkg1_todoList[mtx_index_1] = true;
                }
            }
            for (mtx_index_0 = 0; mtx_index_0 < 16; mtx_index_0++) {
                if (!networkg0_todoList[mtx_index_0]) {
                    if (networkgs0_mtx[mtx_index_0].try_lock()) {
                        if (mtx_index_0 == 15) {
                            for (int p = 0; p <= 128 - 64; p += 64) {
                                _mm256_store_ps(networkgs0_bi + p, _mm256_add_ps(_mm256_load_ps(networkgs0_bi + p), _mm256_load_ps(networkg0_bi + p)));
                                _mm256_store_ps(networkgs0_bi + p + 8, _mm256_add_ps(_mm256_load_ps(networkgs0_bi + p + 8), _mm256_load_ps(networkg0_bi + p + 8)));
                                _mm256_store_ps(networkgs0_bi + p + 16, _mm256_add_ps(_mm256_load_ps(networkgs0_bi + p + 16), _mm256_load_ps(networkg0_bi + p + 16)));
                                _mm256_store_ps(networkgs0_bi + p + 24, _mm256_add_ps(_mm256_load_ps(networkgs0_bi + p + 24), _mm256_load_ps(networkg0_bi + p + 24)));
                                _mm256_store_ps(networkgs0_bi + p + 32, _mm256_add_ps(_mm256_load_ps(networkgs0_bi + p + 32), _mm256_load_ps(networkg0_bi + p + 32)));
                                _mm256_store_ps(networkgs0_bi + p + 40, _mm256_add_ps(_mm256_load_ps(networkgs0_bi + p + 40), _mm256_load_ps(networkg0_bi + p + 40)));
                                _mm256_store_ps(networkgs0_bi + p + 48, _mm256_add_ps(_mm256_load_ps(networkgs0_bi + p + 48), _mm256_load_ps(networkg0_bi + p + 48)));
                                _mm256_store_ps(networkgs0_bi + p + 56, _mm256_add_ps(_mm256_load_ps(networkgs0_bi + p + 56), _mm256_load_ps(networkg0_bi + p + 56)));
                            }
                        }
                        //_mm_prefetch((const char*)(networkg0 + (6272 * mtx_index_0)), _MM_HINT_T0);
                        for (int p = 0; p < 8; p++) {
                            forward_a(0)
                            forward_a(64)
                            forward_a(128)
                            forward_a(192)
                            forward_a(256)
                            forward_a(320)
                            forward_a(384)
                            forward_a(448)
                            forward_a(512)
                            forward_a(576)
                            forward_a(640)
                            forward_a(704)
                            __asm__ volatile (
                                "vmovaps (%0), %%ymm0\n"
                                "vmovaps 0x20(%0), %%ymm1\n"

                                "vmovaps (%1), %%ymm2\n"
                                "vmovaps 0x20(%1), %%ymm3\n"

                                "vaddps %%ymm2, %%ymm0, %%ymm0\n"
                                "vaddps %%ymm3, %%ymm1, %%ymm1\n"

                                "vmovaps %%ymm0, (%1)\n"
                                "vmovaps %%ymm1, 0x20(%1)\n"

                                : 
                                : "r" (networkg0 + ((mtx_index_0 * 8 + p) * 784) + 768), "r" (networkgs0 + ((mtx_index_0 * 8 + p) * 784) + 768)
                                : "memory", "ymm0", "ymm1", "ymm2", "ymm3"
                            );
                        }
                        networkgs0_mtx[mtx_index_0].unlock();
                        networkg0_todoList[mtx_index_0] = true;
                    }
                }
            }
        }
        //Init todoList
        networkg2_todoList[0] = false;
        networkg1_todoList[0] = false;
        for (int i = 0; i < 16; i++) {
            networkg0_todoList[i] = false;
        }

        //Clear Temp
        //_mm_prefetch((const char*)(networkg0), _MM_HINT_T0);
        std::fill(networkg0, networkg0 + 100352, 0);
        std::fill(networkg0_bi, networkg0_bi + 128, 0);
        //_mm_prefetch((const char*)(networkg1), _MM_HINT_T0);
        std::fill(networkg1, networkg1 + 4096, 0);
        std::fill(networkg1_bi, networkg1_bi + 32, 0);
        //_mm_prefetch((const char*)(networkg2), _MM_HINT_T0);
        std::fill(networkg2, networkg2 + 320, 0);
        std::fill(networkg2_bi, networkg2_bi + 10, 0);

        mtx.lock();
        MSETotal += MSError;//Add Lost to Global Cost of this Batch
        reportI++;//This Thread Finished BP
        if (reportI == tpool->TSize) {//If All Threads Finished BP
            //Update Weights & Bias
            //_mm_prefetch((const char*)(networkg0), _MM_HINT_T0);
            memcpy(network0, networkgs0, 401408);
            memcpy(network0_bi, networkgs0_bi, 512);
            //_mm_prefetch((const char*)(networkg1), _MM_HINT_T0);
            memcpy(network1, networkgs1, 16384);
            memcpy(network1_bi, networkgs1_bi, 128);
            //_mm_prefetch((const char*)(networkg2), _MM_HINT_T0);
            memcpy(network2, networkgs2, 1280);
            memcpy(network2_bi, networkgs2_bi, 40);

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

        //rate *= 1.01f;

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

    for (int i = 0; i < 128; ++i) {
        network0_bi[i] = dis(gen);
    }
    for (int i = 0; i < 32; ++i) {
        network1_bi[i] = dis(gen);
    }
    for (int i = 0; i < 10; ++i) {
        network2_bi[i] = dis(gen);
    }

    memcpy(networkgs0, network0, 401408);
    memcpy(networkgs0_bi, network0_bi, 512);
    memcpy(networkgs1, network1, 16384);
    memcpy(networkgs1_bi, network1_bi, 128);
    memcpy(networkgs2, network2, 1280);
    memcpy(networkgs2_bi, network2_bi, 40);

    //Method 2: Load Model from bin
    /*std::ifstream infile("network_data.bin", std::ios::binary);
    if (infile.is_open()) {
        infile.read(reinterpret_cast<char*>(network0), 128 * 785 * sizeof(float));
        infile.read(reinterpret_cast<char*>(network1), 32 * 129 * sizeof(float));
        infile.read(reinterpret_cast<char*>(network2), 10 * 33 * sizeof(float));
        infile.read(reinterpret_cast<char*>(network0_bi), 128 * sizeof(float));
        infile.read(reinterpret_cast<char*>(network1_bi), 32 * sizeof(float));
        infile.read(reinterpret_cast<char*>(network2_bi), 10 * sizeof(float));
        infile.close();
        std::cout << "Model Loaded\n";
    } else {
        std::cerr << "Failed to load model\n";
        return 1;
    }*/

    batchSize = 90;
    std::vector<std::thread> threads = tpool->init(15);//BatchSize must be a positive integer multiple of the number of threads

    rate = 0.001;//Learning Rate
    aim = 0.01;//Aiming Loss(MSE in total)
    dts = 100;//Data Limited(Max 60000)

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


    //Save the model
    /*std::ofstream outfile("network_data.bin", std::ios::binary);
    if (outfile.is_open()) {
        outfile.write(reinterpret_cast<char*>(network0), 128 * 785 * sizeof(float));
        outfile.write(reinterpret_cast<char*>(network1), 32 * 129 * sizeof(float));
        outfile.write(reinterpret_cast<char*>(network2), 10 * 33 * sizeof(float));
        outfile.write(reinterpret_cast<char*>(network0_bi), 128 * sizeof(float));
        outfile.write(reinterpret_cast<char*>(network1_bi), 32 * sizeof(float));
        outfile.write(reinterpret_cast<char*>(network2_bi), 10 * sizeof(float));
        outfile.close();
        std::cout << "Network Model Data Saved\n";
    } else {
        std::cerr << "Failed to save model\n";
    }*/
}