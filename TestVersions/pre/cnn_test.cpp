//g++ -O3 -std=c++20 -march=native -funroll-all-loops -mavx2 -o cnnt.exe cnn_test.cpp

#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <immintrin.h>

#include "cifar10_reader.hpp"


using namespace std;



float(*train_images)[3072] = new float[50000][3072];

float* kernel0 = static_cast<float*>(_mm_malloc(4 * 4 * 3 * 4 * sizeof(float), 32));//每个通道有深度为4的滤波器，卷积核输入4*4大小图像，3个（输入）通道；平铺后该层共12个卷积核；通道格式ABCABC...
float* kernel1 = static_cast<float*>(_mm_malloc(4 * 4 * 4 * 4 * sizeof(float), 32));//每个通道有深度为4的滤波器，卷积核输入4*4大小图像，4个（输入）通道；平铺后该层共12个卷积核；通道格式ABCDABCD





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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.2f, 0.2f);

    /*for (int i = 0; i < ; ++i) {
        // = dis(gen);
    }*/


    int i = 0;
    for (; i < 4*3*4*4; ++i) {
        kernel0[i] = 0;
    }
    for (i = 0; i < 4*4*4*4; ++i) {
        kernel1[i] = 0;
    }

    auto dataset = cifar::read_dataset();
    convert_and_rearrange(dataset.training_images);
    

    float SSum;//For Softmax Activation
    int j, p, q, dtIndex;//Loop Index
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
    float* temp0_f4 = static_cast<float*>(_mm_malloc(4 * sizeof(float), 16));
    float* temp1_f4 = static_cast<float*>(_mm_malloc(4 * sizeof(float), 16));
    alignas(32) const int extract_ACEG_shuffle_mask[8] = { 0, 2, 4, 6, -1, -1, -1, -1 };
    const __m256i extract_ACEG_mask = _mm256_load_si256((const __m256i*)extract_ACEG_shuffle_mask);
    __m256 and_mask_pre = _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFFFFFF));
    alignas(32) const int extract_ACEG_shuffle_mask_128[4] = { 0, 2, -1, -1 };
    const __m128i extract_ACEG_mask_128 = _mm_load_si128((const __m128i*)extract_ACEG_shuffle_mask_128);
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
    float* tempf = static_cast<float*>(_mm_malloc(8 * sizeof(float), 32));
    float* featureMapb0 = static_cast<float*>(_mm_malloc(4 * 289 * sizeof(float), 32));//Pooling Layer 0 Output (Format: ABCDABCD)
    std::fill(featureMapb0, featureMapb0 + 4 * 289, 0);
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
    float* featureMapn0 = static_cast<float*>(_mm_malloc(4 * 289 * sizeof(float), 32));//Activation Layer 0 Output (Format: ABCDABCD)
    std::fill(featureMapn0, featureMapn0 + 4 * 289, 0);
    float* featureMap1 = static_cast<float*>(_mm_malloc(4 * 196 * sizeof(float), 32));//Conv Layer 1 Output (14 * 14 feature) (Accumulation for fully-connected-nn input) (Format: AABBCCDD)
    std::fill(featureMap1, featureMap1 + 4 * 196, 0);


    auto start = std::chrono::high_resolution_clock::now();

    for (int thisDtId = 0; thisDtId < 50000; thisDtId++) {

        //Filter 0
        for (p = 0; p < 29; p++) {
            _mm_prefetch((const char*)(train_images[thisDtId] + (96 * p)), _MM_HINT_T0);
            for (q = 0; q < 29 * 3; q += 3) {
                sum = _mm256_mul_ps(_mm256_load_ps(kernel0), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + q));//96=32*3
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0 + 12), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0 + 24), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0 + 36), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q), sum);
                //处理完整上面不能被8整除的部分
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * p) + q + 8), _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1)));
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 20), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q + 8), sum4);
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 32), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q + 8), sum4);
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 44), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q + 8), sum4);
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
                sum = _mm256_mul_ps(_mm256_load_ps(kernel0 + 48), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + q));//96=32*3
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0 + 48 + 12), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0 + 48 + 24), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0 + 48 + 36), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q), sum);
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 48 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * p) + q + 8), _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1)));
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 48 + 20), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q + 8), sum4);
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 48 + 32), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q + 8), sum4);
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 48 + 44), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q + 8), sum4);
                sum4 = _mm_hadd_ps(sum4, sum4);
                sum4 = _mm_hadd_ps(sum4, sum4);
                featureMap0[1][(30 * p) + (q / 3)] = _mm_cvtss_f32(sum4);
            }
        }
        //Filter 2
        for (p = 0; p < 29; p++) {
            _mm_prefetch((const char*)(train_images[thisDtId] + (96 * p)), _MM_HINT_T0);
            for (q = 0; q < 29 * 3; q += 3) {
                sum = _mm256_mul_ps(_mm256_load_ps(kernel0 + 96), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + q));//96=32*3
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0 + 96 + 12), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0 + 96 + 24), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0 + 96 + 36), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q), sum);
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 96 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * p) + q + 8), _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1)));
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 96 + 20), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q + 8), sum4);
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 96 + 32), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q + 8), sum4);
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 96 + 44), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q + 8), sum4);
                sum4 = _mm_hadd_ps(sum4, sum4);
                sum4 = _mm_hadd_ps(sum4, sum4);
                featureMap0[2][(30 * p) + (q / 3)] = _mm_cvtss_f32(sum4);
            }
        }
        //Filter 3
        for (p = 0; p < 29; p++) {
            _mm_prefetch((const char*)(train_images[thisDtId] + (96 * p)), _MM_HINT_T0);
            for (q = 0; q < 29 * 3; q += 3) {
                sum = _mm256_mul_ps(_mm256_load_ps(kernel0 + 144), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + q));//96=32*3
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0 + 144 + 12), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0 + 144 + 24), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_loadu_ps(kernel0 + 144 + 36), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q), sum);
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 144 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * p) + q + 8), _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1)));
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 144 + 20), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + q + 8), sum4);
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 144 + 32), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + q + 8), sum4);
                sum4 = _mm_fmadd_ps(_mm_load_ps(kernel0 + 144 + 44), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + q + 8), sum4);
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
                _mm256_store_ps(tempf, _mm256_blendv_ps(temp2, temp0, mask));
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
            _mm_store_ps(temp0_f4, _mm_blendv_ps(temp2_128, temp0_128, mask_128));
            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = tempf[0];
            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 2))] = tempf[2];
            _mm_store_ps(temp0_f4, _mm_blendv_ps(temp3_128, temp1_128, mask_128));
            _mm_store_ps(temp1_f4, _mm_blendv_ps(padding0_128, padding1_128, mask_128));
            maxId0[0][(15 * p) + (q / 2)] = temp0_f4[0] + temp1_f4[0];
            maxId0[0][(15 * p) + (q / 2) + 1] = temp0_f4[2] + temp1_f4[2];
            q += 4;

            for (; q <= 30 - 2; q += 2) {
                if (featureMap0[0][(2 * 30 * p) + q] > featureMap0[0][(2 * 30 * (p + 1)) + q]) {
                    if (featureMap0[0][(2 * 30 * p) + q] > featureMap0[0][(2 * 30 * p) + q + 1]) {
                        if (featureMap0[0][(2 * 30 * p) + q] > featureMap0[0][(2 * 30 * (p + 1)) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * p) + q];
                        }
                        else {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[0][(15 * p) + (q / 2)] += 30 + 1;
                        }
                        //featureMapb0[0][(17 * (p + 1)) + (q / 2) + 1] = featureMap0[0][(2 * 30 * p) + q];
                    }
                    else {
                        if (featureMap0[0][(2 * 30 * p) + q + 1] > featureMap0[0][(2 * 30 * (p + 1)) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * p) + q + 1];
                            maxId0[0][(15 * p) + (q / 2)] += 1;
                        }
                        else {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[0][(15 * p) + (q / 2)] += 30 + 1;
                        }
                        //featureMapb0[0][(17 * (p + 1)) + (q / 2) + 2] = featureMap0[0][(2 * 30 * p) + q + 1];
                    }
                }
                else {
                    if (featureMap0[0][(2 * 30 * (p + 1)) + q] > featureMap0[0][(2 * 30 * (p + 1)) + q + 1]) {
                        if (featureMap0[0][(2 * 30 * (p + 1)) + q] > featureMap0[0][(2 * 30 * p) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * (p + 1)) + q];
                            maxId0[0][(15 * p) + (q / 2)] += 30;
                        }
                        else {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[0][(15 * p) + (q / 2)] += 30 + 1;
                        }
                        //featureMapb0[0][(17 * (p + 1)) + (q / 2) + 1] = featureMap0[0][(2 * 30 * (p + 1)) + q];
                    }
                    else {
                        if (featureMap0[0][(2 * 30 * (p + 1)) + q + 1] > featureMap0[0][(2 * 30 * p) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1))] = featureMap0[0][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[0][(15 * p) + (q / 2)] += 30 + 1;
                        }
                        else {
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
                _mm256_store_ps(tempf, _mm256_blendv_ps(temp2, temp0, mask));
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
            _mm_store_ps(temp0_f4, _mm_blendv_ps(temp2_128, temp0_128, mask_128));
            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = tempf[0];
            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 2)) + 1] = tempf[2];
            _mm_store_ps(temp0_f4, _mm_blendv_ps(temp3_128, temp1_128, mask_128));
            _mm_store_ps(temp1_f4, _mm_blendv_ps(padding0_128, padding1_128, mask_128));
            maxId0[1][(15 * p) + (q / 2)] = temp0_f4[0] + temp1_f4[0];
            maxId0[1][(15 * p) + (q / 2) + 1] = temp0_f4[2] + temp1_f4[2];
            q += 4;

            for (; q <= 30 - 2; q += 2) {
                if (featureMap0[1][(2 * 30 * p) + q] > featureMap0[1][(2 * 30 * (p + 1)) + q]) {
                    if (featureMap0[1][(2 * 30 * p) + q] > featureMap0[1][(2 * 30 * p) + q + 1]) {
                        if (featureMap0[1][(2 * 30 * p) + q] > featureMap0[1][(2 * 30 * (p + 1)) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * p) + q];
                        }
                        else {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[1][(15 * p) + (q / 2)] += 30 + 1;
                        }
                    }
                    else {
                        if (featureMap0[1][(2 * 30 * p) + q + 1] > featureMap0[1][(2 * 30 * (p + 1)) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * p) + q + 1];
                            maxId0[1][(15 * p) + (q / 2)] += 1;
                        }
                        else {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[1][(15 * p) + (q / 2)] += 30 + 1;
                        }
                    }
                }
                else {
                    if (featureMap0[1][(2 * 30 * (p + 1)) + q] > featureMap0[1][(2 * 30 * (p + 1)) + q + 1]) {
                        if (featureMap0[1][(2 * 30 * (p + 1)) + q] > featureMap0[1][(2 * 30 * p) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * (p + 1)) + q];
                            maxId0[1][(15 * p) + (q / 2)] += 30;
                        }
                        else {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[1][(15 * p) + (q / 2)] += 30 + 1;
                        }
                    }
                    else {
                        if (featureMap0[1][(2 * 30 * (p + 1)) + q + 1] > featureMap0[1][(2 * 30 * p) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 1] = featureMap0[1][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[1][(15 * p) + (q / 2)] += 30 + 1;
                        }
                        else {
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
                _mm256_store_ps(tempf, _mm256_blendv_ps(temp2, temp0, mask));
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
            _mm_store_ps(temp0_f4, _mm_blendv_ps(temp2_128, temp0_128, mask_128));
            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = tempf[0];
            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 2)) + 2] = tempf[2];
            _mm_store_ps(temp0_f4, _mm_blendv_ps(temp3_128, temp1_128, mask_128));
            _mm_store_ps(temp1_f4, _mm_blendv_ps(padding0_128, padding1_128, mask_128));
            maxId0[2][(15 * p) + (q / 2)] = temp0_f4[0] + temp1_f4[0];
            maxId0[2][(15 * p) + (q / 2) + 1] = temp0_f4[2] + temp1_f4[2];
            q += 4;

            for (; q <= 30 - 2; q += 2) {
                if (featureMap0[2][(2 * 30 * p) + q] > featureMap0[2][(2 * 30 * (p + 1)) + q]) {
                    if (featureMap0[2][(2 * 30 * p) + q] > featureMap0[2][(2 * 30 * p) + q + 1]) {
                        if (featureMap0[2][(2 * 30 * p) + q] > featureMap0[2][(2 * 30 * (p + 1)) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * p) + q];
                        }
                        else {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[2][(15 * p) + (q / 2)] += 30 + 1;
                        }
                    }
                    else {
                        if (featureMap0[2][(2 * 30 * p) + q + 1] > featureMap0[2][(2 * 30 * (p + 1)) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * p) + q + 1];
                            maxId0[2][(15 * p) + (q / 2)] += 1;
                        }
                        else {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[2][(15 * p) + (q / 2)] += 30 + 1;
                        }
                    }
                }
                else {
                    if (featureMap0[2][(2 * 30 * (p + 1)) + q] > featureMap0[2][(2 * 30 * (p + 1)) + q + 1]) {
                        if (featureMap0[2][(2 * 30 * (p + 1)) + q] > featureMap0[2][(2 * 30 * p) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * (p + 1)) + q];
                            maxId0[2][(15 * p) + (q / 2)] += 30;
                        }
                        else {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[2][(15 * p) + (q / 2)] += 30 + 1;
                        }
                    }
                    else {
                        if (featureMap0[2][(2 * 30 * (p + 1)) + q + 1] > featureMap0[2][(2 * 30 * p) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 2] = featureMap0[2][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[2][(15 * p) + (q / 2)] += 30 + 1;
                        }
                        else {
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
                _mm256_store_ps(tempf, _mm256_blendv_ps(temp2, temp0, mask));
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
            _mm_store_ps(temp0_f4, _mm_blendv_ps(temp2_128, temp0_128, mask_128));
            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = tempf[0];
            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 2)) + 3] = tempf[2];
            _mm_store_ps(temp0_f4, _mm_blendv_ps(temp3_128, temp1_128, mask_128));
            _mm_store_ps(temp1_f4, _mm_blendv_ps(padding0_128, padding1_128, mask_128));
            maxId0[3][(15 * p) + (q / 2)] = temp0_f4[0] + temp1_f4[0];
            maxId0[3][(15 * p) + (q / 2) + 1] = temp0_f4[2] + temp1_f4[2];
            q += 4;

            for (; q <= 30 - 2; q += 2) {
                if (featureMap0[3][(2 * 30 * p) + q] > featureMap0[3][(2 * 30 * (p + 1)) + q]) {
                    if (featureMap0[3][(2 * 30 * p) + q] > featureMap0[3][(2 * 30 * p) + q + 1]) {
                        if (featureMap0[3][(2 * 30 * p) + q] > featureMap0[3][(2 * 30 * (p + 1)) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * p) + q];
                        }
                        else {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[3][(15 * p) + (q / 2)] += 30 + 1;
                        }
                    }
                    else {
                        if (featureMap0[3][(2 * 30 * p) + q + 1] > featureMap0[3][(2 * 30 * (p + 1)) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * p) + q + 1];
                            maxId0[3][(15 * p) + (q / 2)] += 1;
                        }
                        else {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[3][(15 * p) + (q / 2)] += 30 + 1;
                        }
                    }
                }
                else {
                    if (featureMap0[3][(2 * 30 * (p + 1)) + q] > featureMap0[3][(2 * 30 * (p + 1)) + q + 1]) {
                        if (featureMap0[3][(2 * 30 * (p + 1)) + q] > featureMap0[3][(2 * 30 * p) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * (p + 1)) + q];
                            maxId0[3][(15 * p) + (q / 2)] += 30;
                        }
                        else {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[3][(15 * p) + (q / 2)] += 30 + 1;
                        }
                    }
                    else {
                        if (featureMap0[3][(2 * 30 * (p + 1)) + q + 1] > featureMap0[3][(2 * 30 * p) + q + 1]) {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * (p + 1)) + q + 1];
                            maxId0[3][(15 * p) + (q / 2)] += 30 + 1;
                        }
                        else {
                            featureMapb0[(4 * 17 * (p + 1)) + (4 * ((q / 2) + 1)) + 3] = featureMap0[3][(2 * 30 * p) + q + 1];
                            maxId0[3][(15 * p) + (q / 2)] += 1;
                        }
                    }
                }
            }
        }


        //Activation Leaky-Relu
        for (i = 0; i <= (289 * 4) - 8; i += 8) {
            temp0 = _mm256_load_ps(featureMapb0 + i);
            _mm256_store_ps(featureMapn0 + i, _mm256_mul_ps(temp0, _mm256_blendv_ps(padding1, leakyRelu_alpha, _mm256_cmp_ps(temp0, _mm256_setzero_ps(), _CMP_LT_OQ))));
        }
        for (i = 0; i <= (289 * 4) - 4; i += 4) {
            temp0_128 = _mm_load_ps(featureMapb0 + i);
            _mm_store_ps(featureMapn0 + i, _mm_mul_ps(temp0_128, _mm_blendv_ps(padding1_128, leakyRelu_alpha_128, _mm_cmp_ps(temp0_128, _mm_setzero_ps(), _CMP_LT_OQ))));
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
                sum = _mm256_mul_ps(_mm256_load_ps(kernel1), _mm256_load_ps(featureMapn0 + (68 * p) + q));
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 8), _mm256_load_ps(featureMapn0 + (68 * p) + q + 8), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 16), _mm256_load_ps(featureMapn0 + (68 * (p + 1)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 24), _mm256_load_ps(featureMapn0 + (68 * (p + 1)) + q + 8), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 32), _mm256_load_ps(featureMapn0 + (68 * (p + 2)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 40), _mm256_load_ps(featureMapn0 + (68 * (p + 2)) + q + 8), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 48), _mm256_load_ps(featureMapn0 + (68 * (p + 3)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 56), _mm256_load_ps(featureMapn0 + (68 * (p + 3)) + q + 8), sum);
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
                sum = _mm256_mul_ps(_mm256_load_ps(kernel1 + 48), _mm256_load_ps(featureMapn0 + (68 * p) + q));
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 48 + 8), _mm256_load_ps(featureMapn0 + (68 * p) + q + 8), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 48 + 16), _mm256_load_ps(featureMapn0 + (68 * (p + 1)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 48 + 24), _mm256_load_ps(featureMapn0 + (68 * (p + 1)) + q + 8), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 48 + 32), _mm256_load_ps(featureMapn0 + (68 * (p + 2)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 48 + 40), _mm256_load_ps(featureMapn0 + (68 * (p + 2)) + q + 8), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 48 + 48), _mm256_load_ps(featureMapn0 + (68 * (p + 3)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 48 + 56), _mm256_load_ps(featureMapn0 + (68 * (p + 3)) + q + 8), sum);
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
                sum = _mm256_mul_ps(_mm256_load_ps(kernel1 + 96), _mm256_load_ps(featureMapn0 + (68 * p) + q));
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 96 + 8), _mm256_load_ps(featureMapn0 + (68 * p) + q + 8), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 96 + 16), _mm256_load_ps(featureMapn0 + (68 * (p + 1)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 96 + 24), _mm256_load_ps(featureMapn0 + (68 * (p + 1)) + q + 8), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 96 + 32), _mm256_load_ps(featureMapn0 + (68 * (p + 2)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 96 + 40), _mm256_load_ps(featureMapn0 + (68 * (p + 2)) + q + 8), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 96 + 48), _mm256_load_ps(featureMapn0 + (68 * (p + 3)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 96 + 56), _mm256_load_ps(featureMapn0 + (68 * (p + 3)) + q + 8), sum);
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
                sum = _mm256_mul_ps(_mm256_load_ps(kernel1 + 144), _mm256_load_ps(featureMapn0 + (68 * p) + q));
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 144 + 8), _mm256_load_ps(featureMapn0 + (68 * p) + q + 8), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 144 + 16), _mm256_load_ps(featureMapn0 + (68 * (p + 1)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 144 + 24), _mm256_load_ps(featureMapn0 + (68 * (p + 1)) + q + 8), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 144 + 32), _mm256_load_ps(featureMapn0 + (68 * (p + 2)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 144 + 40), _mm256_load_ps(featureMapn0 + (68 * (p + 2)) + q + 8), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 144 + 48), _mm256_load_ps(featureMapn0 + (68 * (p + 3)) + q), sum);
                sum = _mm256_fmadd_ps(_mm256_load_ps(kernel1 + 144 + 56), _mm256_load_ps(featureMapn0 + (68 * (p + 3)) + q + 8), sum);
                sum4 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
                sum4 = _mm_hadd_ps(sum4, sum4);
                sum4 = _mm_hadd_ps(sum4, sum4);
                featureMap1[588 + (14 * p) + (q / 4)] = _mm_cvtss_f32(sum4);
            }
        }


    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << duration.count() << "s" << std::endl;

    return 0;
}



/*
* （附）卷积核扩展技术：用于当卷积核总浮点宽度不足于SIMD指令要求时
* 
*   float(*kernel0_expand)[24 * 4] = new float[4][24 * 4]{ 0 };//24为(4*3)和8的最小公倍数，即(4*3)至少展开多少倍后才能达到8的倍数，除非已经超出输入图像宽度
    float* temp_row0 = static_cast<float*>(_mm_malloc(32 * 3 * sizeof(float), 32));
    for (int filterId = 0; filterId < 4; filterId++) {
        for (int lineId = 0; lineId < 4; lineId++) {
            memcpy(&kernel0_expand[filterId][24 * lineId], &kernel0[filterId][4 * 3 * lineId], 4 * (4 * 3));
            memcpy(&kernel0_expand[filterId][24 * lineId + 12], &kernel0[filterId][4 * 3 * lineId], 4 * (4 * 3));
        }
    }
*
*                   _mm_prefetch((const char*)(train_images[thisDtId] + (96 * p)), _MM_HINT_T0);
                    //Step 0
                    for (i = 0; i <= 96 - 24; i += 24) {//展开后的长度（24）能在图像中（96）完整展开填充n次
                        for (j = 0; j < 24; j += 8) {
                            _mm256_storeu_ps(temp_row0 + i + j,
                                _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 72 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + i + j),//Line 3
                                    _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 48 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + i + j),//Line 2
                                        _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 24 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + i + j),//Line 1
                                            _mm256_mul_ps(_mm256_loadu_ps(kernel0_expand[0] + 0 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + i + j))//Line 0
                                        )
                                    )
                                )
                            );
                        }
                    }
                    for (j = 0; j <= 96 - 12; j += 12) {
                        sum4 = _mm_add_ps(
                            _mm_add_ps(_mm256_extractf128_ps(_mm256_loadu_ps(temp_row0 + j), 0), _mm256_extractf128_ps(_mm256_loadu_ps(temp_row0 + j), 1)),
                            _mm_loadu_ps(temp_row0 + j + 8)
                        );
                        sum4 = _mm_hadd_ps(sum4, sum4);
                        sum4 = _mm_hadd_ps(sum4, sum4);
                        featureMap0[0][(30 * p) + (j / 3)] += _mm_cvtss_f32(sum4);
                    }

                    _mm_prefetch((const char*)(train_images[thisDtId] + (96 * p)), _MM_HINT_T0);
                    //Step 1
                    for (i = 3; i <= 96 - 24; i += 24) {
                        for (j = 0; j < 24; j += 8) {
                            _mm256_storeu_ps(temp_row0 + i + j,
                                _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 72 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + i + j),//Line 3
                                    _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 48 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + i + j),//Line 2
                                        _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 24 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + i + j),//Line 1
                                            _mm256_mul_ps(_mm256_loadu_ps(kernel0_expand[0] + 0 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + i + j))//Line 0
                                        )
                                    )
                                )
                            );
                        }
                    }
                    //处理剩余不能铺满24的部分，尝试补一个12（即一个卷积核宽）
                    for (; i <= 96 - 8; i += 8) {
                        _mm256_storeu_ps(temp_row0 + i,
                            _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 72), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + i),//Line 3
                                _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 48), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + i),//Line 2
                                    _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 24), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + i),//Line 1
                                        _mm256_mul_ps(_mm256_loadu_ps(kernel0_expand[0] + 0), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + i))//Line 0
                                    )
                                )
                            )
                        );
                    }
                    for (; i <= 96 - 4; i += 4) {
                        _mm_storeu_ps(temp_row0 + i,
                            _mm_fmadd_ps(_mm_loadu_ps(kernel0_expand[0] + 72 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + i),//Line 3
                                _mm_fmadd_ps(_mm_loadu_ps(kernel0_expand[0] + 48 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + i),//Line 2
                                    _mm_fmadd_ps(_mm_loadu_ps(kernel0_expand[0] + 24 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + i),//Line 1
                                        _mm_mul_ps(_mm_loadu_ps(kernel0_expand[0] + 0 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * p) + i))//Line 0
                                    )
                                )
                            )
                        );
                    }
                    for (j = 3; j <= 96 - 12; j += 12) {
                        sum4 = _mm_add_ps(
                            _mm_add_ps(_mm256_extractf128_ps(_mm256_loadu_ps(temp_row0 + j), 0), _mm256_extractf128_ps(_mm256_loadu_ps(temp_row0 + j), 1)),
                            _mm_loadu_ps(temp_row0 + j + 8)
                        );
                        sum4 = _mm_hadd_ps(sum4, sum4);
                        sum4 = _mm_hadd_ps(sum4, sum4);
                        featureMap0[0][(30 * p) + (j / 3)] += _mm_cvtss_f32(sum4);
                    }

                    _mm_prefetch((const char*)(train_images[thisDtId] + (96 * p)), _MM_HINT_T0);
                    //Step 2
                    for (i = 6; i <= 96 - 24; i += 24) {
                        for (j = 0; j < 24; j += 8) {
                            _mm256_storeu_ps(temp_row0 + i + j,
                                _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 72 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + i + j),//Line 3
                                    _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 48 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + i + j),//Line 2
                                        _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 24 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + i + j),//Line 1
                                            _mm256_mul_ps(_mm256_loadu_ps(kernel0_expand[0] + 0 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + i + j))//Line 0
                                        )
                                    )
                                )
                            );
                        }
                    }
                    for (; i <= 96 - 8; i += 8) {
                        _mm256_storeu_ps(temp_row0 + i,
                            _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 72), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + i),//Line 3
                                _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 48), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + i),//Line 2
                                    _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 24), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + i),//Line 1
                                        _mm256_mul_ps(_mm256_loadu_ps(kernel0_expand[0] + 0), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + i))//Line 0
                                    )
                                )
                            )
                        );
                    }
                    for (; i <= 96 - 4; i += 4) {
                        _mm_storeu_ps(temp_row0 + i,
                            _mm_fmadd_ps(_mm_loadu_ps(kernel0_expand[0] + 72 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + i),//Line 3
                                _mm_fmadd_ps(_mm_loadu_ps(kernel0_expand[0] + 48 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + i),//Line 2
                                    _mm_fmadd_ps(_mm_loadu_ps(kernel0_expand[0] + 24 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + i),//Line 1
                                        _mm_mul_ps(_mm_loadu_ps(kernel0_expand[0] + 0 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * p) + i))//Line 0
                                    )
                                )
                            )
                        );
                    }
                    for (; i <= 96 - 1; i++) {
                        temp_row0[i] = (kernel0_expand[0][i] * train_images[thisDtId][(96 * p) + i])
                            + (kernel0_expand[0][24 + i]) * (train_images[thisDtId][(96 * (p + 1)) + i])
                            + (kernel0_expand[0][48 + i]) * (train_images[thisDtId][(96 * (p + 2)) + i])
                            + (kernel0_expand[0][72 + i]) * (train_images[thisDtId][(96 * (p + 3)) + i]);
                    }
                    for (j = 6; j <= 96 - 12; j += 12) {
                        sum4 = _mm_add_ps(
                            _mm_add_ps(_mm256_extractf128_ps(_mm256_loadu_ps(temp_row0 + j), 0), _mm256_extractf128_ps(_mm256_loadu_ps(temp_row0 + j), 1)),
                            _mm_loadu_ps(temp_row0 + j + 8)
                        );
                        sum4 = _mm_hadd_ps(sum4, sum4);
                        sum4 = _mm_hadd_ps(sum4, sum4);
                        featureMap0[0][(30 * p) + (j / 3)] += _mm_cvtss_f32(sum4);
                    }

                    _mm_prefetch((const char*)(train_images[thisDtId] + (96 * p)), _MM_HINT_T0);
                    //Step 3
                    for (i = 9; i <= 96 - 24; i += 24) {
                        for (j = 0; j < 24; j += 8) {
                            _mm256_storeu_ps(temp_row0 + i + j,
                                _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 72 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + i + j),//Line 3
                                    _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 48 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + i + j),//Line 2
                                        _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 24 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + i + j),//Line 1
                                            _mm256_mul_ps(_mm256_loadu_ps(kernel0_expand[0] + 0 + j), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + i + j))//Line 0
                                        )
                                    )
                                )
                            );
                        }
                    }
                    for (; i <= 96 - 8; i += 8) {
                        _mm256_storeu_ps(temp_row0 + i,
                            _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 72), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + i),//Line 3
                                _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 48), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + i),//Line 2
                                    _mm256_fmadd_ps(_mm256_loadu_ps(kernel0_expand[0] + 24), _mm256_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + i),//Line 1
                                        _mm256_mul_ps(_mm256_loadu_ps(kernel0_expand[0] + 0), _mm256_loadu_ps(train_images[thisDtId] + (96 * p) + i))//Line 0
                                    )
                                )
                            )
                        );
                    }
                    for (; i <= 96 - 4; i += 4) {
                        _mm_storeu_ps(temp_row0 + i,
                            _mm_fmadd_ps(_mm_loadu_ps(kernel0_expand[0] + 72 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 3)) + i),//Line 3
                                _mm_fmadd_ps(_mm_loadu_ps(kernel0_expand[0] + 48 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 2)) + i),//Line 2
                                    _mm_fmadd_ps(_mm_loadu_ps(kernel0_expand[0] + 24 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * (p + 1)) + i),//Line 1
                                        _mm_mul_ps(_mm_loadu_ps(kernel0_expand[0] + 0 + 8), _mm_loadu_ps(train_images[thisDtId] + (96 * p) + i))//Line 0
                                    )
                                )
                            )
                        );
                    }
                    for (j = 9; j <= 96 - 12; j += 12) {
                        sum4 = _mm_add_ps(
                            _mm_add_ps(_mm256_extractf128_ps(_mm256_loadu_ps(temp_row0 + j), 0), _mm256_extractf128_ps(_mm256_loadu_ps(temp_row0 + j), 1)),
                            _mm_loadu_ps(temp_row0 + j + 8)
                        );
                        sum4 = _mm_hadd_ps(sum4, sum4);
                        sum4 = _mm_hadd_ps(sum4, sum4);
                        featureMap0[0][(30 * p) + (j / 3)] += _mm_cvtss_f32(sum4);
                    }
*/
