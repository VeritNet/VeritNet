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
#include <cassert>
#include <immintrin.h>

#include "cifar10_reader.hpp"


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
TP* tpool = new TP;
int dts;

float(*train_images)[3072] = new float[50000][3072];

float(*kernel0)[4 * 3 * 4] = new float[4][4 * 3 * 4]{ 0 };//每个通道有深度为4的滤波器，卷积核输入4*4大小图像，3个（输入）通道；平铺后该层共12个卷积核；通道格式ABCABC...
float(*kernel1)[4 * 4 * 4] = new float[4][4 * 4 * 4]{ 0 };//每个通道有深度为4的滤波器，卷积核输入4*4大小图像，4个（输入）通道；平铺后该层共12个卷积核；通道格式ABCDABCD

float* network0 = static_cast<float*>(_mm_malloc(128 * 784 * sizeof(float), 32));
float* network1 = static_cast<float*>(_mm_malloc(32 * 128 * sizeof(float), 32));
float* network2 = static_cast<float*>(_mm_malloc(10 * 32 * sizeof(float), 32));
float* network0_bi = static_cast<float*>(_mm_malloc(128 * sizeof(float), 32));
float* network1_bi = static_cast<float*>(_mm_malloc(32 * sizeof(float), 32));
float* network2_bi = static_cast<float*>(_mm_malloc(10 * sizeof(float), 32));

float(*train_labels)[10] = new float[50000][10];

int batchSize;
float MSETotal;
//Network Gradients (Shared Memory)
float* networkgs0 = static_cast<float*>(_mm_malloc(128 * 784 * sizeof(float), 32));
float* networkgs1 = static_cast<float*>(_mm_malloc(32 * 128 * sizeof(float), 32));
float* networkgs2 = static_cast<float*>(_mm_malloc(10 * 32 * sizeof(float), 32));
float* networkgs0_bi = static_cast<float*>(_mm_malloc(128 * sizeof(float), 32));
float* networkgs1_bi = static_cast<float*>(_mm_malloc(32 * sizeof(float), 32));
float* networkgs2_bi = static_cast<float*>(_mm_malloc(10 * sizeof(float), 32));


//Locks for Blocks
std::vector<std::mutex> networkgs0_mtx(8);
std::vector<std::mutex> networkgs1_mtx(1);
std::vector<std::mutex> networkgs2_mtx(1);

float(*kernelgs0)[4 * 4 * 3] = new float[4][4 * 4 * 3];
float(*kernelgs1)[4 * 4 * 4] = new float[4][4 * 4 * 4];

bool gate;
int reportI = 0;
int BId = 0;//Batch Id
float rate, aim, err;
inline void trainNet(int TId) {
    tpool->TGate[TId] = new std::counting_semaphore<1>(0);
    int thisDtId;


    //Network Gradients (Thread Memory)
    float* networkg0 = static_cast<float*>(_mm_malloc(128 * 784 * sizeof(float), 32));//Network Gradients (Thread Memory)
    float* networkg0_bi = static_cast<float*>(_mm_malloc(128 * sizeof(float), 32));
    float* networkb0 = static_cast<float*>(_mm_malloc(128 * sizeof(float), 32));//w*x + b (Before Activation) for hidden layer 0
    float* networkn0 = static_cast<float*>(_mm_malloc(128 * sizeof(float), 32));//Activation for hidden layer 0

    float* networkg1 = static_cast<float*>(_mm_malloc(32 * 128 * sizeof(float), 32));//Network Gradients (Thread Memory)
    float* networkg1_bi = static_cast<float*>(_mm_malloc(32 * sizeof(float), 32));
    float* networkb1 = static_cast<float*>(_mm_malloc(32 * sizeof(float), 32));//..for hidden layer 1
    float* networkn1 = static_cast<float*>(_mm_malloc(32 * sizeof(float), 32));//..for hidden layer 1

    float* networkg2 = static_cast<float*>(_mm_malloc(10 * 32 * sizeof(float), 32));//Network Gradients (Thread Memory)
    float* networkg2_bi = static_cast<float*>(_mm_malloc(10 * sizeof(float), 32));
    float* networkb2 = static_cast<float*>(_mm_malloc(10 * sizeof(float), 32));//..for hidden layer 2
    float* networkn2 = static_cast<float*>(_mm_malloc(10 * sizeof(float), 32));//..for hidden layer 2

    float(*kernelg0)[4 * 4 * 3] = new float[4][4 * 4 * 3]{ 0 };
    float(*kernelg1)[4 * 4 * 4] = new float[4][4 * 4 * 4]{ 0 };

    //Block update of gradients in shared memory
    vector<bool> networkg0_todoList(8);
    vector<bool> networkg1_todoList(1);
    vector<bool> networkg2_todoList(1);

    vector<bool> kernelg_todoList(1);

    float MSError{};//Error Sum of All Data in this Epoch in this Thread

    float SSum;//For Softmax Activation
    int i, j, p, q, dtIndex;//Loop Index
    int mtx_index_0, mtx_index_1, mtx_index_2;//Lock Index
    __m256 sum, factor, mask, temp0, temp1, temp2, temp3, temp4,
        exp_temp, exp_fx, exp_floor_fx, exp_x_squared;//For SIMD
    __m256 padding0 = _mm256_set1_ps(0);
    __m256 padding1 = _mm256_set1_ps(1);
    __m256 padding30 = _mm256_set1_ps(30);
    __m128 padding0_128 = _mm_set1_ps(0);
    __m128 padding1_128 = _mm_set1_ps(1);
    __m128 padding30_128 = _mm_set1_ps(30);
    __m128 sum_high, sum_low, sum4, mask_128, temp0_128, temp1_128, temp2_128, temp3_128;//For SIMD
    float temp0_f4[4], temp1_f4[4];
    alignas(32) const int extract_ACEG_shuffle_mask[8] = { 0, 2, 4, 6, -1, -1, -1, -1 };
    const __m256i extract_ACEG_mask = _mm256_loadu_si256((const __m256i*)extract_ACEG_shuffle_mask);
    __m256 and_mask_pre = _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFFFFFF));
    alignas(32) const int extract_ACEG_shuffle_mask_128[4] = { 0, 2, -1, -1 };
    const __m128i extract_ACEG_mask_128 = _mm_loadu_si128((const __m128i*)extract_ACEG_shuffle_mask_128);
    __m128 and_mask_pre_128 = _mm_castsi128_ps(_mm_set1_epi32(0xFFFFFFFF));
    const int temp_increase[4] = { 0,2,4,6 };

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
    const __m256 padding_half = _mm256_set1_ps(0.5f);
    const __m256i INT_7F = _mm256_set1_epi32(0x7F);

    float(*featureMap0)[900] = new float[4][900]{ 0 };//Conv Layer 0 Output (29 * 29 feature + 1 padding) (Format: AABBCCDD)
    float tempf[8];
    float* featureMapb0 = new float[4 * 289] { 0 };//Pooling Layer 0 Output (Format: ABCDABCD)
    //float(*featureMapb0)[289] = new float[4][289]{ 0 };//Pooling Layer 0 Output
    float maxId_init[255];
    for (p = 0; p < 15; p++) {
        for (q = 0; q <= 15; q++) {
            maxId_init[15 * p + q] = (2 * 2 * 15 * p) + (2 * q);//2 line / 2 pixels (in featureMap0) a group
        }
    }
    float(*maxId0)[225] = new float[4][225]{ 0 };
    memcpy(maxId0[0], maxId_init, 900);//225*4
    memcpy(maxId0[1], maxId_init, 900);//225*4
    memcpy(maxId0[2], maxId_init, 900);//225*4
    memcpy(maxId0[3], maxId_init, 900);//225*4
    __m256 leakyRelu_alpha = _mm256_set1_ps(0.01f);
    __m128 leakyRelu_alpha_128 = _mm_set1_ps(0.01f);
    float* featureMapn0 = new float[4 * 289] { 0 };//Activation Layer 0 Output (Format: ABCDABCD)

    float* featureMap1 = static_cast<float*>(_mm_malloc(4 * 196 * sizeof(float), 32));//Conv Layer 1 Output (14 * 14 feature) (Accumulation for fully-connected-nn input) (Format: AABBCCDD)

    //The gradient of neurons is equal to the gradient of neuron bias
    float* networkg0_neuron = static_cast<float*>(_mm_malloc(128 * sizeof(float), 32));
    float* networkg1_neuron = static_cast<float*>(_mm_malloc(32 * sizeof(float), 32));
    float* networkg2_neuron = static_cast<float*>(_mm_malloc(10 * sizeof(float), 32));

    std::fill(networkg0_neuron, networkg0_neuron + 128, 0);
    std::fill(networkg1_neuron, networkg1_neuron + 32, 0);
    std::fill(networkg2_neuron, networkg2_neuron + 10, 0);

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


    for (;;) {
        tpool->TGate[TId]->acquire();


        for (dtIndex = batchSize / tpool->TSize - 1; dtIndex >= 0; dtIndex--) {//Train all data in this task
            thisDtId = (batchSize * BId) + (TId * (batchSize / tpool->TSize)) + dtIndex;

            //Filter 0
            for (p = 0; p < 29; p++) {
                _mm_prefetch((const char*)(train_images[thisDtId] + (96 * p)), _MM_HINT_T0);
                for (q = 0; q < 29 * 3; q += 3) {
                    sum = _mm256_mul_ps(_mm256_loadu_ps(kernel0[0]), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + q));//96=32*3
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0[0] + 12), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0[0] + 24), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0[0] + 36), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q), sum);
                    //处理完整上面不能被8整除的部分
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[0] + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * p) + q + 8), _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1)));
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[0] + 20), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q + 8), sum4);
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[0] + 32), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q + 8), sum4);
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[0] + 44), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q + 8), sum4);
                    //规约
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    featureMap0[0][(30 * p) + (q / 3)] = _mm_cvtss_f32(sum4);
                }
            }
            //Filter 1
            for (p = 0; p < 29; p++) {
                _mm_prefetch((const char*)(train_images[thisDtId] + (96 * p)), _MM_HINT_T0);
                for (q = 0; q < 29 * 3; q += 3) {
                    sum = _mm256_mul_ps(_mm256_loadu_ps(kernel0[1]), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + q));//96=32*3
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0[1] + 12), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0[1] + 24), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0[1] + 36), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q), sum);
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[1] + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * p) + q + 8), _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1)));
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[1] + 20), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q + 8), sum4);
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[1] + 32), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q + 8), sum4);
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[1] + 44), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q + 8), sum4);
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    featureMap0[1][(30 * p) + (q / 3)] = _mm_cvtss_f32(sum4);
                }
            }
            //Filter 2
            for (p = 0; p < 29; p++) {
                _mm_prefetch((const char*)(train_images[thisDtId] + (96 * p)), _MM_HINT_T0);
                for (q = 0; q < 29 * 3; q += 3) {
                    sum = _mm256_mul_ps(_mm256_loadu_ps(kernel0[2]), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + q));//96=32*3
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0[2] + 12), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0[2] + 24), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0[2] + 36), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q), sum);
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[2] + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * p) + q + 8), _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1)));
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[2] + 20), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q + 8), sum4);
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[2] + 32), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q + 8), sum4);
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[2] + 44), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q + 8), sum4);
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    featureMap0[2][(30 * p) + (q / 3)] = _mm_cvtss_f32(sum4);
                }
            }
            //Filter 3
            for (p = 0; p < 29; p++) {
                _mm_prefetch((const char*)(train_images[thisDtId] + (96 * p)), _MM_HINT_T0);
                for (q = 0; q < 29 * 3; q += 3) {
                    sum = _mm256_mul_ps(_mm256_loadu_ps(kernel0[3]), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + q));//96=32*3
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0[3] + 12), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0[3] + 24), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0[3] + 36), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q), sum);
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[3] + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * p) + q + 8), _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1)));
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[3] + 20), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q + 8), sum4);
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[3] + 32), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q + 8), sum4);
                    sum4 = _mm_fmadd_ps(_mm_loadu_ps(kernel0[3] + 44), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q + 8), sum4);
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    featureMap0[3][(30 * p) + (q / 3)] = _mm_cvtss_f32(sum4);
                }
            }

            //Max Pooling
            for (p = 0; p < 15; p++) {
                q = 0;
                for (; q <= 30 - 8; q += 8) {
                    temp0 = _mm256_loadu_ps(featureMap0[0] + (2 * 30 * p) + q);
                    temp1 = _mm256_loadu_ps(featureMap0[0] + (2 * 30 * (p + 1)) + q);
                    mask = _mm256_and_ps(_mm256_cmp_ps(temp0, temp1, _CMP_LT_OQ), and_mask_pre);//vertical mask
                    temp2 = _mm256_blendv_ps(temp0, temp1, mask);//vertical max
                    temp3 = _mm256_blendv_ps(padding0, padding30, mask);//vertical maxId
                    temp0 = _mm256_permute_ps(temp2, 0x39);//shifted max
                    temp1 = _mm256_permute_ps(temp3, 0x39);//shifted maxId
                    mask = _mm256_and_ps(_mm256_cmp_ps(temp2, temp0, _CMP_LT_OQ), and_mask_pre);//horizontal mask
                    _mm256_storeu_ps(tempf, _mm256_blendv_ps(temp2, temp0, mask));
                    featureMapb0[(4 * 17 * (p + 1/*1 padding*/)) + (4 * ((q / 2) + 1/*1 padding*/))] = tempf[0];
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 2))] = tempf[2];
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 3))] = tempf[4];
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 4))] = tempf[6];
                    //_mm_storeu_ps(featureMapb0[0] + (17 * (p + 1)) + (q / 2) + 1, _mm256_castps256_ps128(_mm256_permutevar8x32_ps(_mm256_blendv_ps(temp2, temp0, mask), extract_ACEG_mask)));//horizontal max
                    _mm_storeu_ps(maxId0[0] + (15 * p) + (q / 2), _mm_add_ps(_mm256_castps256_ps128(_mm256_permutevar8x32_ps(_mm256_blendv_ps(temp3, temp1, mask), extract_ACEG_mask)), _mm256_castps256_ps128(_mm256_permutevar8x32_ps(_mm256_blendv_ps(padding0, padding1, mask), extract_ACEG_mask))));//final maxId
                }
                temp0_128 = _mm_loadu_ps(featureMap0[0] + (2 * 30 * p) + q);
                temp1_128 = _mm_loadu_ps(featureMap0[0] + (2 * 30 * (p + 1)) + q);
                mask_128 = _mm_and_ps(_mm_cmp_ps(temp0_128, temp1_128, _CMP_LT_OQ), and_mask_pre_128);//vertical mask
                temp2_128 = _mm_blendv_ps(temp0_128, temp1_128, mask_128);//vertical max
                temp3_128 = _mm_blendv_ps(padding0_128, padding30_128, mask_128);//vertical maxId
                temp0_128 = _mm_permute_ps(temp2_128, 0x39);//shifted max
                temp1_128 = _mm_permute_ps(temp3_128, 0x39);//shifted maxId
                mask_128 = _mm_and_ps(_mm_cmp_ps(temp2_128, temp0_128, _CMP_LT_OQ), and_mask_pre_128);//horizontal mask
                _mm_storeu_ps(temp0_f4, _mm_blendv_ps(temp2_128, temp0_128, mask_128));
                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = tempf[0];
                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 2))] = tempf[2];
                _mm_storeu_ps(temp0_f4, _mm_blendv_ps(temp3_128, temp1_128, mask_128));
                _mm_storeu_ps(temp1_f4, _mm_blendv_ps(padding0_128, padding1_128, mask_128));
                maxId0[0][(15 * p) + (q / 2)] = temp0_f4[0] + temp1_f4[0];
                maxId0[0][(15 * p) + (q / 2) + 1] = temp0_f4[2] + temp1_f4[2];
                q += 4;

                for (; q <= 30 - 2; q += 2) {
                    if (featureMap0[0][(2 * 30 * p) + q] > featureMap0[0][(2 * 30 * (p + 1)) + q]) {
                        if (featureMap0[0][(2 * 30 * p) + q] > featureMap0[0][(2 * 30 * p) + q + 1]) {
                            if (featureMap0[0][(2 * 30 * p) + q] > featureMap0[0][(2 * 30 * (p + 1)) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * p) + q];
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[0][(15 * p) + (q / 2)] += 30 + 1;
                            }
                            //featureMapb0[0][(17 * (p + 1)) + (q / 2) + 1] = featureMap0[0][(2 * 30 * p) + q];
                        } else {
                            if (featureMap0[0][(2 * 30 * p) + q + 1] > featureMap0[0][(2 * 30 * (p + 1)) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * p) + q + 1];
                                maxId0[0][(15 * p) + (q / 2)] += 1;
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[0][(15 * p) + (q / 2)] += 30 + 1;
                            }
                            //featureMapb0[0][(17 * (p + 1)) + (q / 2) + 2] = featureMap0[0][(2 * 30 * p) + q + 1];
                        }
                    } else {
                        if (featureMap0[0][(2 * 30 * (p + 1)) + q] > featureMap0[0][(2 * 30 * (p + 1)) + q + 1]) {
                            if (featureMap0[0][(2 * 30 * (p + 1)) + q] > featureMap0[0][(2 * 30 * p) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * (p + 1)) + q];
                                maxId0[0][(15 * p) + (q / 2)] += 30;
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[0][(15 * p) + (q / 2)] += 30 + 1;
                            }
                            //featureMapb0[0][(17 * (p + 1)) + (q / 2) + 1] = featureMap0[0][(2 * 30 * (p + 1)) + q];
                        } else {
                            if (featureMap0[0][(2 * 30 * (p + 1)) + q + 1] > featureMap0[0][(2 * 30 * p) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[0][(15 * p) + (q / 2)] += 30 + 1;
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * p) + q + 1];
                                maxId0[0][(15 * p) + (q / 2)] += 1;
                            }
                            //featureMapb0[0][(17 * (p + 1)) + (q / 2) + 2] = featureMap0[0][(2 * 30 * (p + 1)) + q + 1];
                        }
                    }
                }
            }
            //1
            for (p = 0; p < 15; p++) {
                q = 0;
                for (; q <= 30 - 8; q += 8) {
                    temp0 = _mm256_loadu_ps(featureMap0[1] + (2 * 30 * p) + q);
                    temp1 = _mm256_loadu_ps(featureMap0[1] + (2 * 30 * (p + 1)) + q);
                    mask = _mm256_and_ps(_mm256_cmp_ps(temp0, temp1, _CMP_LT_OQ), and_mask_pre);//vertical mask
                    temp2 = _mm256_blendv_ps(temp0, temp1, mask);//vertical max
                    temp3 = _mm256_blendv_ps(padding0, padding30, mask);//vertical maxId
                    temp0 = _mm256_permute_ps(temp2, 0x39);//shifted max
                    temp1 = _mm256_permute_ps(temp3, 0x39);//shifted maxId
                    mask = _mm256_and_ps(_mm256_cmp_ps(temp2, temp0, _CMP_LT_OQ), and_mask_pre);//horizontal mask
                    _mm256_storeu_ps(tempf, _mm256_blendv_ps(temp2, temp0, mask));
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = tempf[0];
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 2)) + 1] = tempf[2];
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 3)) + 1] = tempf[4];
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 4)) + 1] = tempf[6];
                    _mm_storeu_ps(maxId0[1] + (15 * p) + (q / 2), _mm_add_ps(_mm256_castps256_ps128(_mm256_permutevar8x32_ps(_mm256_blendv_ps(temp3, temp1, mask), extract_ACEG_mask)), _mm256_castps256_ps128(_mm256_permutevar8x32_ps(_mm256_blendv_ps(padding0, padding1, mask), extract_ACEG_mask))));//final maxId
                }
                temp0_128 = _mm_loadu_ps(featureMap0[1] + (2 * 30 * p) + q);
                temp1_128 = _mm_loadu_ps(featureMap0[1] + (2 * 30 * (p + 1)) + q);
                mask_128 = _mm_and_ps(_mm_cmp_ps(temp0_128, temp1_128, _CMP_LT_OQ), and_mask_pre_128);//vertical mask
                temp2_128 = _mm_blendv_ps(temp0_128, temp1_128, mask_128);//vertical max
                temp3_128 = _mm_blendv_ps(padding0_128, padding30_128, mask_128);//vertical maxId
                temp0_128 = _mm_permute_ps(temp2_128, 0x39);//shifted max
                temp1_128 = _mm_permute_ps(temp3_128, 0x39);//shifted maxId
                mask_128 = _mm_and_ps(_mm_cmp_ps(temp2_128, temp0_128, _CMP_LT_OQ), and_mask_pre_128);//horizontal mask
                _mm_storeu_ps(temp0_f4, _mm_blendv_ps(temp2_128, temp0_128, mask_128));
                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = tempf[0];
                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 2)) + 1] = tempf[2];
                _mm_storeu_ps(temp0_f4, _mm_blendv_ps(temp3_128, temp1_128, mask_128));
                _mm_storeu_ps(temp1_f4, _mm_blendv_ps(padding0_128, padding1_128, mask_128));
                maxId0[1][(15 * p) + (q / 2)] = temp0_f4[0] + temp1_f4[0];
                maxId0[1][(15 * p) + (q / 2) + 1] = temp0_f4[2] + temp1_f4[2];
                q += 4;

                for (; q <= 30 - 2; q += 2) {
                    if (featureMap0[1][(2 * 30 * p) + q] > featureMap0[1][(2 * 30 * (p + 1)) + q]) {
                        if (featureMap0[1][(2 * 30 * p) + q] > featureMap0[1][(2 * 30 * p) + q + 1]) {
                            if (featureMap0[1][(2 * 30 * p) + q] > featureMap0[1][(2 * 30 * (p + 1)) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * p) + q];
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[1][(15 * p) + (q / 2)] += 30 + 1;
                            }
                        }  else {
                            if (featureMap0[1][(2 * 30 * p) + q + 1] > featureMap0[1][(2 * 30 * (p + 1)) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * p) + q + 1];
                                maxId0[1][(15 * p) + (q / 2)] += 1;
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[1][(15 * p) + (q / 2)] += 30 + 1;
                            }
                        }
                    } else {
                        if (featureMap0[1][(2 * 30 * (p + 1)) + q] > featureMap0[1][(2 * 30 * (p + 1)) + q + 1]) {
                            if (featureMap0[1][(2 * 30 * (p + 1)) + q] > featureMap0[1][(2 * 30 * p) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * (p + 1)) + q];
                                maxId0[1][(15 * p) + (q / 2)] += 30;
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[1][(15 * p) + (q / 2)] += 30 + 1;
                            }
                        } else {
                            if (featureMap0[1][(2 * 30 * (p + 1)) + q + 1] > featureMap0[1][(2 * 30 * p) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[1][(15 * p) + (q / 2)] += 30 + 1;
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * p) + q + 1];
                                maxId0[1][(15 * p) + (q / 2)] += 1;
                            }
                        }
                    }
                }
            }
            //2
            for (p = 0; p < 15; p++) {
                q = 0;
                for (; q <= 30 - 8; q += 8) {
                    temp0 = _mm256_loadu_ps(featureMap0[2] + (2 * 30 * p) + q);
                    temp1 = _mm256_loadu_ps(featureMap0[2] + (2 * 30 * (p + 1)) + q);
                    mask = _mm256_and_ps(_mm256_cmp_ps(temp0, temp1, _CMP_LT_OQ), and_mask_pre);//vertical mask
                    temp2 = _mm256_blendv_ps(temp0, temp1, mask);//vertical max
                    temp3 = _mm256_blendv_ps(padding0, padding30, mask);//vertical maxId
                    temp0 = _mm256_permute_ps(temp2, 0x39);//shifted max
                    temp1 = _mm256_permute_ps(temp3, 0x39);//shifted maxId
                    mask = _mm256_and_ps(_mm256_cmp_ps(temp2, temp0, _CMP_LT_OQ), and_mask_pre);//horizontal mask
                    _mm256_storeu_ps(tempf, _mm256_blendv_ps(temp2, temp0, mask));
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = tempf[0];
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 2)) + 2] = tempf[2];
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 3)) + 2] = tempf[4];
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 4)) + 2] = tempf[6];
                    _mm_storeu_ps(maxId0[2] + (15 * p) + (q / 2), _mm_add_ps(_mm256_castps256_ps128(_mm256_permutevar8x32_ps(_mm256_blendv_ps(temp3, temp1, mask), extract_ACEG_mask)), _mm256_castps256_ps128(_mm256_permutevar8x32_ps(_mm256_blendv_ps(padding0, padding1, mask), extract_ACEG_mask))));//final maxId
                }
                temp0_128 = _mm_loadu_ps(featureMap0[2] + (2 * 30 * p) + q);
                temp1_128 = _mm_loadu_ps(featureMap0[2] + (2 * 30 * (p + 1)) + q);
                mask_128 = _mm_and_ps(_mm_cmp_ps(temp0_128, temp1_128, _CMP_LT_OQ), and_mask_pre_128);//vertical mask
                temp2_128 = _mm_blendv_ps(temp0_128, temp1_128, mask_128);//vertical max
                temp3_128 = _mm_blendv_ps(padding0_128, padding30_128, mask_128);//vertical maxId
                temp0_128 = _mm_permute_ps(temp2_128, 0x39);//shifted max
                temp1_128 = _mm_permute_ps(temp3_128, 0x39);//shifted maxId
                mask_128 = _mm_and_ps(_mm_cmp_ps(temp2_128, temp0_128, _CMP_LT_OQ), and_mask_pre_128);//horizontal mask
                _mm_storeu_ps(temp0_f4, _mm_blendv_ps(temp2_128, temp0_128, mask_128));
                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = tempf[0];
                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 2)) + 2] = tempf[2];
                _mm_storeu_ps(temp0_f4, _mm_blendv_ps(temp3_128, temp1_128, mask_128));
                _mm_storeu_ps(temp1_f4, _mm_blendv_ps(padding0_128, padding1_128, mask_128));
                maxId0[2][(15 * p) + (q / 2)] = temp0_f4[0] + temp1_f4[0];
                maxId0[2][(15 * p) + (q / 2) + 1] = temp0_f4[2] + temp1_f4[2];
                q += 4;

                for (; q <= 30 - 2; q += 2) {
                    if (featureMap0[2][(2 * 30 * p) + q] > featureMap0[2][(2 * 30 * (p + 1)) + q]) {
                        if (featureMap0[2][(2 * 30 * p) + q] > featureMap0[2][(2 * 30 * p) + q + 1]) {
                            if (featureMap0[2][(2 * 30 * p) + q] > featureMap0[2][(2 * 30 * (p + 1)) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * p) + q];
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[2][(15 * p) + (q / 2)] += 30 + 1;
                            }
                        } else {
                            if (featureMap0[2][(2 * 30 * p) + q + 1] > featureMap0[2][(2 * 30 * (p + 1)) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * p) + q + 1];
                                maxId0[2][(15 * p) + (q / 2)] += 1;
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[2][(15 * p) + (q / 2)] += 30 + 1;
                            }
                        }
                    } else {
                        if (featureMap0[2][(2 * 30 * (p + 1)) + q] > featureMap0[2][(2 * 30 * (p + 1)) + q + 1]) {
                            if (featureMap0[2][(2 * 30 * (p + 1)) + q] > featureMap0[2][(2 * 30 * p) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * (p + 1)) + q];
                                maxId0[2][(15 * p) + (q / 2)] += 30;
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[2][(15 * p) + (q / 2)] += 30 + 1;
                            }
                        } else {
                            if (featureMap0[2][(2 * 30 * (p + 1)) + q + 1] > featureMap0[2][(2 * 30 * p) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[2][(15 * p) + (q / 2)] += 30 + 1;
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * p) + q + 1];
                                maxId0[2][(15 * p) + (q / 2)] += 1;
                            }
                        }
                    }
                }
            }
            //3
            for (p = 0; p < 15; p++) {
                q = 0;
                for (; q <= 30 - 8; q += 8) {
                    temp0 = _mm256_loadu_ps(featureMap0[3] + (2 * 30 * p) + q);
                    temp1 = _mm256_loadu_ps(featureMap0[3] + (2 * 30 * (p + 1)) + q);
                    mask = _mm256_and_ps(_mm256_cmp_ps(temp0, temp1, _CMP_LT_OQ), and_mask_pre);//vertical mask
                    temp2 = _mm256_blendv_ps(temp0, temp1, mask);//vertical max
                    temp3 = _mm256_blendv_ps(padding0, padding30, mask);//vertical maxId
                    temp0 = _mm256_permute_ps(temp2, 0x39);//shifted max
                    temp1 = _mm256_permute_ps(temp3, 0x39);//shifted maxId
                    mask = _mm256_and_ps(_mm256_cmp_ps(temp2, temp0, _CMP_LT_OQ), and_mask_pre);//horizontal mask
                    _mm256_storeu_ps(tempf, _mm256_blendv_ps(temp2, temp0, mask));
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = tempf[0];
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 2)) + 3] = tempf[2];
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 3)) + 3] = tempf[4];
                    featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 4)) + 3] = tempf[6];
                    _mm_storeu_ps(maxId0[3] + (15 * p) + (q / 2), _mm_add_ps(_mm256_castps256_ps128(_mm256_permutevar8x32_ps(_mm256_blendv_ps(temp3, temp1, mask), extract_ACEG_mask)), _mm256_castps256_ps128(_mm256_permutevar8x32_ps(_mm256_blendv_ps(padding0, padding1, mask), extract_ACEG_mask))));//final maxId
                }
                temp0_128 = _mm_loadu_ps(featureMap0[3] + (2 * 30 * p) + q);
                temp1_128 = _mm_loadu_ps(featureMap0[3] + (2 * 30 * (p + 1)) + q);
                mask_128 = _mm_and_ps(_mm_cmp_ps(temp0_128, temp1_128, _CMP_LT_OQ), and_mask_pre_128);//vertical mask
                temp2_128 = _mm_blendv_ps(temp0_128, temp1_128, mask_128);//vertical max
                temp3_128 = _mm_blendv_ps(padding0_128, padding30_128, mask_128);//vertical maxId
                temp0_128 = _mm_permute_ps(temp2_128, 0x39);//shifted max
                temp1_128 = _mm_permute_ps(temp3_128, 0x39);//shifted maxId
                mask_128 = _mm_and_ps(_mm_cmp_ps(temp2_128, temp0_128, _CMP_LT_OQ), and_mask_pre_128);//horizontal mask
                _mm_storeu_ps(temp0_f4, _mm_blendv_ps(temp2_128, temp0_128, mask_128));
                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = tempf[0];
                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 2)) + 3] = tempf[2];
                _mm_storeu_ps(temp0_f4, _mm_blendv_ps(temp3_128, temp1_128, mask_128));
                _mm_storeu_ps(temp1_f4, _mm_blendv_ps(padding0_128, padding1_128, mask_128));
                maxId0[3][(15 * p) + (q / 2)] = temp0_f4[0] + temp1_f4[0];
                maxId0[3][(15 * p) + (q / 2) + 1] = temp0_f4[2] + temp1_f4[2];
                q += 4;

                for (; q <= 30 - 2; q += 2) {
                    if (featureMap0[3][(2 * 30 * p) + q] > featureMap0[3][(2 * 30 * (p + 1)) + q]) {
                        if (featureMap0[3][(2 * 30 * p) + q] > featureMap0[3][(2 * 30 * p) + q + 1]) {
                            if (featureMap0[3][(2 * 30 * p) + q] > featureMap0[3][(2 * 30 * (p + 1)) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * p) + q];
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[3][(15 * p) + (q / 2)] += 30 + 1;
                            }
                        } else {
                            if (featureMap0[3][(2 * 30 * p) + q + 1] > featureMap0[3][(2 * 30 * (p + 1)) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * p) + q + 1];
                                maxId0[3][(15 * p) + (q / 2)] += 1;
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[3][(15 * p) + (q / 2)] += 30 + 1;
                            }
                        }
                    } else {
                        if (featureMap0[3][(2 * 30 * (p + 1)) + q] > featureMap0[3][(2 * 30 * (p + 1)) + q + 1]) {
                            if (featureMap0[3][(2 * 30 * (p + 1)) + q] > featureMap0[3][(2 * 30 * p) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * (p + 1)) + q];
                                maxId0[3][(15 * p) + (q / 2)] += 30;
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[3][(15 * p) + (q / 2)] += 30 + 1;
                            }
                        } else {
                            if (featureMap0[3][(2 * 30 * (p + 1)) + q + 1] > featureMap0[3][(2 * 30 * p) + q + 1]) {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * (p + 1)) + q + 1];
                                maxId0[3][(15 * p) + (q / 2)] += 30 + 1;
                            } else {
                                featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * p) + q + 1];
                                maxId0[3][(15 * p) + (q / 2)] += 1;
                            }
                        }
                    }
                }
            }


            //Activation Leaky-Relu
            for (i = 0; i <= (289 * 4) - 8; i += 8) {
                temp0 = _mm256_loadu_ps(featureMapb0 + i);
                _mm256_storeu_ps(featureMapn0 + i, _mm256_mul_ps(temp0, _mm256_blendv_ps(padding1, leakyRelu_alpha, _mm256_cmp_ps(temp0, _mm256_setzero_ps(), _CMP_LT_OQ))));
            }
            for (i = 0; i <= (289 * 4) - 4; i += 4) {
                temp0_128 = _mm_loadu_ps(featureMapb0 + i);
                _mm_storeu_ps(featureMapn0 + i, _mm_mul_ps(temp0_128, _mm_blendv_ps(padding1_128, leakyRelu_alpha_128, _mm_cmp_ps(temp0_128, _mm_setzero_ps(), _CMP_LT_OQ))));
            }
            /*for (; i < (289 * 4); i++) {
                if (featureMapb0[i] < 0) {
                    featureMapn0[i] *= 0.01f;
                }
            }*/

            //Conv layer 1
            //Filter 0
            for (p = 0; p < 14; p++) {
                _mm_prefetch((const char*)(featureMapn0 + (68 * p)), _MM_HINT_T0);
                for (q = 0; q < 14 * 4; q += 4) {
                    sum = _mm256_mul_ps(_mm256_loadu_ps(kernel1[0]), _mm256_loadu_ps(featureMapn0 + (68 * p) + q));
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[0] + 8), _mm256_loadu_ps(featureMapn0 + (68 * p) + q + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[0] + 16), _mm256_loadu_ps(featureMapn0 + (68 * (p + 1)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[0] + 24), _mm256_loadu_ps(featureMapn0 + (68 * (p + 1)) + q + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[0] + 32), _mm256_loadu_ps(featureMapn0 + (68 * (p + 2)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[0] + 40), _mm256_loadu_ps(featureMapn0 + (68 * (p + 2)) + q + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[0] + 48), _mm256_loadu_ps(featureMapn0 + (68 * (p + 3)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[0] + 56), _mm256_loadu_ps(featureMapn0 + (68 * (p + 3)) + q + 8), sum);
                    sum4 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    featureMap1[(14 * p) + (q / 4)] = _mm_cvtss_f32(sum4);
                }
            }
            //Filter 1
            for (p = 0; p < 14; p++) {
                _mm_prefetch((const char*)(featureMapn0 + (68 * p)), _MM_HINT_T0);
                for (q = 0; q < 14 * 4; q += 4) {
                    sum = _mm256_mul_ps(_mm256_loadu_ps(kernel1[1]), _mm256_loadu_ps(featureMapn0 + (68 * p) + q));
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[1] + 8), _mm256_loadu_ps(featureMapn0 + (68 * p) + q + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[1] + 16), _mm256_loadu_ps(featureMapn0 + (68 * (p + 1)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[1] + 24), _mm256_loadu_ps(featureMapn0 + (68 * (p + 1)) + q + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[1] + 32), _mm256_loadu_ps(featureMapn0 + (68 * (p + 2)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[1] + 40), _mm256_loadu_ps(featureMapn0 + (68 * (p + 2)) + q + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[1] + 48), _mm256_loadu_ps(featureMapn0 + (68 * (p + 3)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[1] + 56), _mm256_loadu_ps(featureMapn0 + (68 * (p + 3)) + q + 8), sum);
                    sum4 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    featureMap1[196 + (14 * p) + (q / 4)] = _mm_cvtss_f32(sum4);
                }
            }
            //Filter 2
            for (p = 0; p < 14; p++) {
                _mm_prefetch((const char*)(featureMapn0 + (68 * p)), _MM_HINT_T0);
                for (q = 0; q < 14 * 4; q += 4) {
                    sum = _mm256_mul_ps(_mm256_loadu_ps(kernel1[2]), _mm256_loadu_ps(featureMapn0 + (68 * p) + q));
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[2] + 8), _mm256_loadu_ps(featureMapn0 + (68 * p) + q + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[2] + 16), _mm256_loadu_ps(featureMapn0 + (68 * (p + 1)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[2] + 24), _mm256_loadu_ps(featureMapn0 + (68 * (p + 1)) + q + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[2] + 32), _mm256_loadu_ps(featureMapn0 + (68 * (p + 2)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[2] + 40), _mm256_loadu_ps(featureMapn0 + (68 * (p + 2)) + q + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[2] + 48), _mm256_loadu_ps(featureMapn0 + (68 * (p + 3)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[2] + 56), _mm256_loadu_ps(featureMapn0 + (68 * (p + 3)) + q + 8), sum);
                    sum4 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    featureMap1[392 + (14 * p) + (q / 4)] = _mm_cvtss_f32(sum4);
                }
            }
            //Filter 3
            for (p = 0; p < 14; p++) {
                _mm_prefetch((const char*)(featureMapn0 + (68 * p)), _MM_HINT_T0);
                for (q = 0; q < 14 * 4; q += 4) {
                    sum = _mm256_mul_ps(_mm256_loadu_ps(kernel1[3]), _mm256_loadu_ps(featureMapn0 + (68 * p) + q));
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[3] + 8), _mm256_loadu_ps(featureMapn0 + (68 * p) + q + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[3] + 16), _mm256_loadu_ps(featureMapn0 + (68 * (p + 1)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[3] + 24), _mm256_loadu_ps(featureMapn0 + (68 * (p + 1)) + q + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[3] + 32), _mm256_loadu_ps(featureMapn0 + (68 * (p + 2)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[3] + 40), _mm256_loadu_ps(featureMapn0 + (68 * (p + 2)) + q + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[3] + 48), _mm256_loadu_ps(featureMapn0 + (68 * (p + 3)) + q), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel1[3] + 56), _mm256_loadu_ps(featureMapn0 + (68 * (p + 3)) + q + 8), sum);
                    sum4 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    sum4 = _mm_hadd_ps(sum4, sum4);
                    featureMap1[588 + (14 * p) + (q / 4)] = _mm_cvtss_f32(sum4);
                }
            }


            //Feed Forward
            //Input Layer - Hidden Layer 0
            for (p = 0; p < 128; p++) {
                sum = _mm256_setzero_ps();
                i = 0;
                _mm_prefetch((const char*)(network0 + (p * 784)), _MM_HINT_T0);
                for (; i <= 784 - 64; i += 64) {
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i), _mm256_load_ps(featureMap1 + i), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 8), _mm256_load_ps(featureMap1 + i + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 16), _mm256_load_ps(featureMap1 + i + 16), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 24), _mm256_load_ps(featureMap1 + i + 24), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 32), _mm256_load_ps(featureMap1 + i + 32), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 40), _mm256_load_ps(featureMap1 + i + 40), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 48), _mm256_load_ps(featureMap1 + i + 48), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 56), _mm256_load_ps(featureMap1 + i + 56), sum);
                }
                for (; i <= 784 - 16; i += 16) {
                    _mm_prefetch((const char*)(network0 + i + 16), _MM_HINT_T0);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i), _mm256_load_ps(featureMap1 + i), sum);
                    sum = _mm256_fmadd_ps(_mm256_load_ps(network0 + (p * 784) + i + 8), _mm256_load_ps(featureMap1 + i + 8), sum);
                }
                sum_high = _mm256_extractf128_ps(sum, 1);
                sum_low = _mm256_extractf128_ps(sum, 0);
                sum_low = _mm_add_ps(sum_low, sum_high);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                networkb0[p] = _mm_cvtss_f32(sum_low);
            }
            for (p = 0; p <= 128 - 64; p += 64) {
                _mm256_store_ps(networkb0 + p, _mm256_add_ps(_mm256_load_ps(networkb0 + p), _mm256_load_ps(network0_bi + p)));
                _mm256_store_ps(networkb0 + p + 8, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 8), _mm256_load_ps(network0_bi + p + 8)));
                _mm256_store_ps(networkb0 + p + 16, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 16), _mm256_load_ps(network0_bi + p + 16)));
                _mm256_store_ps(networkb0 + p + 24, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 24), _mm256_load_ps(network0_bi + p + 24)));
                _mm256_store_ps(networkb0 + p + 32, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 32), _mm256_load_ps(network0_bi + p + 32)));
                _mm256_store_ps(networkb0 + p + 40, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 40), _mm256_load_ps(network0_bi + p + 40)));
                _mm256_store_ps(networkb0 + p + 48, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 48), _mm256_load_ps(network0_bi + p + 48)));
                _mm256_store_ps(networkb0 + p + 56, _mm256_add_ps(_mm256_load_ps(networkb0 + p + 56), _mm256_load_ps(network0_bi + p + 56)));
            }
            for (p = 0; p < 128; p++) {
                if (networkb0[p] >= 0) {
                    networkn0[p] = networkb0[p];
                }
                else {
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
            }
            _mm256_store_ps(networkb1, _mm256_add_ps(_mm256_load_ps(networkb1), _mm256_load_ps(network1_bi)));
            _mm256_store_ps(networkb1 + 8, _mm256_add_ps(_mm256_load_ps(networkb1 + 8), _mm256_load_ps(network1_bi + 8)));
            _mm256_store_ps(networkb1 + 16, _mm256_add_ps(_mm256_load_ps(networkb1 + 16), _mm256_load_ps(network1_bi + 16)));
            _mm256_store_ps(networkb1 + 24, _mm256_add_ps(_mm256_load_ps(networkb1 + 24), _mm256_load_ps(network1_bi + 24)));
            for (p = 0; p < 32; p++) {
                if (networkb1[p] >= 0) {
                    networkn1[p] = networkb1[p];
                }
                else {
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
            for (p = 0; p < 10; p++) {
                if (std::isnan(networkn2[p])) {
                    std::cerr << "NaN detected at index " << thisDtId << " " << p << std::endl;
                    for (int a = 0; a < 10; a++) {
                        std::cout << networkb2[a] << ",";
                    }
                    std::cout << std::endl;
                    for (int a = 0; a < 32; a++) {
                        std::cout << networkn1[a] << ",";
                    }
                    assert(false);  // Trigger assertion failure
                }
                MSError += ((train_labels[thisDtId][p] - networkn2[p]) * (train_labels[thisDtId][p] - networkn2[p]));
            }

            //Back Propagation
            //Output Layer - Hidden Layer 2
            for (p = 0; p < 10; p++) {
                networkg2_neuron[p] = -rate * (networkn2[p] - train_labels[thisDtId][p]);
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
            for (p = 0; p < 32; p++) {
                networkg1_neuron[p] = 0;
                for (q = 0; q < 10; q++) {
                    networkg1_neuron[p] += networkg2_neuron[q] * network2[(q * 32) + p];
                }
                if (networkb1[p] < 0) {
                    networkg1_neuron[p] *= exp(networkb1[p]);
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
            for (p = 0; p < 128; p++) {
                networkg0_neuron[p] = 0;
                for (q = 0; q < 32; q++) {
                    networkg0_neuron[p] += networkg1_neuron[q] * network1[(q * 128) + p];
                }
                if (networkb0[p] < 0) {
                    networkg0_neuron[p] *= exp(networkb0[p]);
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
                        : "r" (featureMap1 + i), "r" (networkg0 + (p * 784) + i), "x" (factor)
                        : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", "ymm12", "ymm13", "ymm14"
                    );
                }
            }
            for (p = 0; p < 128; p += 64) {
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
            _mm_prefetch((const char*)(networkn0), _MM_HINT_T0);
            std::fill(networkn0, networkn0 + 128, 0);
            std::fill(networkn1, networkn1 + 32, 0);
            std::fill(networkn2, networkn2 + 10, 0);
            std::fill(networkb0, networkb0 + 128, 0);
            std::fill(networkb1, networkb1 + 32, 0);
            std::fill(networkb2, networkb2 + 10, 0);

            memcpy(maxId0[0], maxId_init, 900);//225*4
            memcpy(maxId0[1], maxId_init, 900);//225*4
            memcpy(maxId0[2], maxId_init, 900);//225*4
            memcpy(maxId0[3], maxId_init, 900);//225*4
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
                    _mm256_store_ps(networkgs2_bi, _mm256_add_ps(_mm256_load_ps(networkgs2_bi), _mm256_load_ps(networkg2_bi)));
                    networkgs2_bi[8] += networkg2_bi[8];
                    networkgs2_bi[9] += networkg2_bi[9];
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
                    _mm256_store_ps(networkgs1_bi, _mm256_add_ps(_mm256_load_ps(networkgs1_bi), _mm256_load_ps(networkg1_bi)));
                    _mm256_store_ps(networkgs1_bi + 8, _mm256_add_ps(_mm256_load_ps(networkgs1_bi + 8), _mm256_load_ps(networkg1_bi + 8)));
                    _mm256_store_ps(networkgs1_bi + 16, _mm256_add_ps(_mm256_load_ps(networkgs1_bi + 16), _mm256_load_ps(networkg1_bi + 16)));
                    _mm256_store_ps(networkgs1_bi + 24, _mm256_add_ps(_mm256_load_ps(networkgs1_bi + 24), _mm256_load_ps(networkg1_bi + 24)));
                    networkgs1_mtx[mtx_index_1].unlock();
                    networkg1_todoList[mtx_index_1] = true;
                }
            }
            for (mtx_index_0 = 0; mtx_index_0 < 8; mtx_index_0++) {
                if (!networkg0_todoList[mtx_index_0]) {
                    if (networkgs0_mtx[mtx_index_0].try_lock()) {
                        if (mtx_index_0 == 7) {
                            for (p = 0; p <= 128 - 64; p += 64) {
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
                        _mm_prefetch((const char*)(networkg0 + (12544 * mtx_index_0)), _MM_HINT_T0);
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
        std::fill(networkg0_bi, networkg0_bi + 128, 0);
        _mm_prefetch((const char*)(networkg1), _MM_HINT_T0);
        std::fill(networkg1, networkg1 + 4096, 0);
        std::fill(networkg1_bi, networkg1_bi + 32, 0);
        _mm_prefetch((const char*)(networkg2), _MM_HINT_T0);
        std::fill(networkg2, networkg2 + 320, 0);
        std::fill(networkg2_bi, networkg2_bi + 10, 0);

        mtx.lock();
        MSETotal += MSError;//Add Lost to Global Cost of this Batch
        reportI++;//This Thread Finished BP
        if (reportI == tpool->TSize) {//If All Threads Finished BP
            //Update Weights & Bias
            _mm_prefetch((const char*)(networkg0), _MM_HINT_T0);
            memcpy(network0, networkgs0, 401408);
            memcpy(network0_bi, networkgs0_bi, 512);
            _mm_prefetch((const char*)(networkg1), _MM_HINT_T0);
            memcpy(network1, networkgs1, 16384);
            memcpy(network1_bi, networkgs1_bi, 128);
            _mm_prefetch((const char*)(networkg2), _MM_HINT_T0);
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
        for (c = 0; c < 50000 / batchSize; c++) {
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

        rate *= 1.01f;

        if (err <= aim) {
            std::cout << "------------------------------" << std::endl;
            std::cout << ">>> finished " << dts * i << " steps (" << i << " Epoch) gradient descent (Cost: " << err << ")" << std::endl;
            break;
        }
        else {
            std::cout << "Epoch " << i << " | Time " << duration.count() << " | MSE " << err << "  " << err / dts << " | rate " << rate << std::endl;
        }
    }
}



inline void convert_and_rearrange(const std::vector<std::vector<uint8_t>>& image) {
    //image.size()==60000;image[n].size()==3072
    for (int img = 0; img < 50000; img++) {
        for (int i = 0; i < 1024; i++) {
            train_images[img][i * 3] = static_cast<int>(image[img][i]) / 255.0f;
            train_images[img][i * 3 + 1] = static_cast<int>(image[img][1024 + i]) / 255.0f;
            train_images[img][i * 3 + 2] = static_cast<int>(image[img][2048 + i]) / 255.0f;
        }
    }
}



int main() {
    std::cout << "1/3 Init Data" << endl;
    gate = true;
    MSETotal = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.2f, 0.2f);
    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 784; ++j) {
            network0[i * 784 + j] = dis(gen);
        }
    }
    for (int i = 0; i < 30; ++i) {
        for (int j = 0; j < 128; ++j) {
            network1[i * 128 + j] = dis(gen);
        }
    }
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 32; ++j) {
            network2[i * 32 + j] = dis(gen);
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


    int i = 0;
    for (; i < 48; ++i) {
        kernel0[0][i] = 0;
    }
    kernel0[0][0] = 2;
    kernel0[0][3] = 3;
    kernel0[0][6] = 2;
    kernel0[0][36] = -2;
    kernel0[0][39] = -3;
    kernel0[0][42] = -2;

    kernel0[1][0] = 2;
    kernel0[1][3] = 3;
    kernel0[1][6] = 2;
    kernel0[1][36] = -2;
    kernel0[1][39] = -3;
    kernel0[1][42] = -2;

    kernel0[2][0] = 2;
    kernel0[2][3] = 3;
    kernel0[2][6] = 2;
    kernel0[2][36] = -2;
    kernel0[2][39] = -3;
    kernel0[2][42] = -2;

    kernel0[3][0] = 2;
    kernel0[3][3] = 3;
    kernel0[3][6] = 2;
    kernel0[3][36] = -2;
    kernel0[3][39] = -3;
    kernel0[3][42] = -2;

    kernel1[0][0] = -1;
    kernel1[0][4] = -1;
    kernel1[0][8] = -1;
    kernel1[0][16] = -1;
    kernel1[0][20] = 8;
    kernel1[0][24] = -1;
    kernel1[0][32] = -1;
    kernel1[0][36] = -1;
    kernel1[0][40] = -1;


    batchSize = 50;
    std::vector<std::thread> threads = tpool->init(1);//BatchSize必须是线程数的正整数倍

    std::cout << "2/3 Load Training Data" << endl;

    auto dataset = cifar::read_dataset();
    convert_and_rearrange(dataset.training_images);
    

    std::cout << "3/3 Ready" << endl;
    train(rate, aim);

    for (int i = 0; i < threads.size(); i++) {
        threads[i].detach();
    }
}
