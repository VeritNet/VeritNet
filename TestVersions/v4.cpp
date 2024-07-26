/*
* g++ -O3 -march=native -funroll-loops -mavx2 -o v4.exe v4.cpp
* Test Version 2024.7.26.4
* [128 Elu, 32 Elu, 10 Softmax]
* Notice: Ubuntu20.04 is not supported, please update to Ubuntu24 or compile in Windows
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

#include "loadMNIST.h"


using namespace std;


std::mutex mtx;//Global Mutex Lock
//Thread Pool
class TP {
public:
    deque<atomic<bool>> TGate;//Wake Conditon
    deque<atomic<bool>> TFree;//Is Free
    vector/*Threads*/< vector<vector<float>> > TData0;//Input Data
    vector/*Threads*/< vector<vector<float>> > TData1;//Label
    int TSize;//Num of threads
    void init(int size);//Create & Detach
    inline void add(vector<vector<float>>& inputDt0, vector<vector<float>>& inputDt1) {//Add Task
        bool noBreak = true;
        while (noBreak) {
            for (int i = 0; i < TSize; i++) {//Find a Free Thread
                if (TFree[i].load()) {
                    TFree[i].store(false);
                    mtx.lock();
                    //Give Task
                    TData0[i] = inputDt0;
                    TData1[i] = inputDt1;
                    mtx.unlock();
                    TGate[i].store(true);//Wake
                    noBreak = false;
                    break;
                }
            }
        }
    }
};
TP* tpool = new TP;//thread pool
int dts;//Training Data Limit
//Weight (and bias at the end of each array) (Shared Memory)
float(*network0)[784 + 1] = new float[128][784 + 1];
float(*network1)[128 + 1] = new float[32][128 + 1];
float(*network2)[32 + 1] = new float[10][32 + 1];
int batchSize;//Batch Size
float MSETotal;//MSE Cost
//Network Gradients (Shared Memory)
float(*networkgs0)[784 + 1] = new float[128][784 + 1];
float(*networkgs1)[128 + 1] = new float[32][128 + 1];
float(*networkgs2)[32 + 1] = new float[10][32 + 1];
//Locks for Blocks
std::vector<std::mutex> networkgs0_mtx(16);
std::vector<std::mutex> networkgs1_mtx(4);
std::vector<std::mutex> networkgs2_mtx(1);
bool gate;//Main Thread Gate
int reportI = 0;
float rate, aim, err;//Learning Rate, MSE aim, Cost of 1 Epoch
inline void trainNet(int TId/*Thread Id*/) {
    //Network Gradients (Thread Memory)
    float(*networkg0)[784 + 1] = new float[128][784 + 1]{};
    float(*networkg1)[128 + 1] = new float[32][128 + 1]{};
    float(*networkg2)[32 + 1] = new float[10][32 + 1]{};
    //Block update of gradients in shared memory
    vector<bool> networkg0_todoList(16);
    vector<bool> networkg1_todoList(4);
    vector<bool> networkg2_todoList(1);
    
    float MSError{};//Error Sum of All Data in this Epoch in this Thread

    float SSum;//For Softmax Activation
    int i, p, q, dtIndex;//Loop Index
    int mtx_index_0, mtx_index_1, mtx_index_2;//Lock Index
    __m256 sum, factor;//For SIMD
    __m128 sum_high, sum_low;//For SIMD

    float networkn0[128] = { 0 };//Activation for hidden layer 0
    float networkn1[32] = { 0 };//..for hidden layer 1
    float networkn2[10] = { 0 };//..for hidden layer 2
    float networkb0[128] = { 0 };//w*x + b (Before Activation) for hidden layer 0
    float networkb1[32] = { 0 };//..for hidden layer 1
    float networkb2[10] = { 0 };//..for hidden layer 2

    //The gradient of neurons is equal to the gradient of neuron bias
    float networkg0_neuron[128] = { 0 };
    float networkg1_neuron[32] = { 0 };
    float networkg2_neuron[10] = { 0 };


    for (;;) {
        while (tpool->TGate[TId].load()==false) {}//Wait
        tpool->TGate[TId].store(false);//Working
        for (dtIndex = tpool->TData0[TId].size() - 1; dtIndex >= 0; dtIndex--) {//Train all data in this task
            //Feed Forward
            //Input Layer - Hidden Layer 0
            for (p = 0; p < 128; p++) {
                sum = _mm256_setzero_ps();
                i = 0;
                for (; i <= 784 - 64; i += 64) {
                    _mm_prefetch((const char*)(network0[p] + i + 64), _MM_HINT_T0);
                    _mm_prefetch((const char*)(tpool->TData0[TId][dtIndex].data() + i + 64), _MM_HINT_T0);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network0[p] + i), _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network0[p] + i + 8), _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network0[p] + i + 16), _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 16), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network0[p] + i + 24), _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 24), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network0[p] + i + 32), _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 32), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network0[p] + i + 40), _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 40), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network0[p] + i + 48), _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 48), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network0[p] + i + 56), _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 56), sum);
                }
                for (; i <= 784 - 16; i += 16) {
                    _mm_prefetch((const char*)(network0[p] + i + 16), _MM_HINT_T0);
                    _mm_prefetch((const char*)(tpool->TData0[TId][dtIndex].data() + i + 16), _MM_HINT_T0);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network0[p] + i), _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network0[p] + i + 8), _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 8), sum);
                }
                sum_high = _mm256_extractf128_ps(sum, 1);
                sum_low = _mm256_extractf128_ps(sum, 0);
                sum_low = _mm_add_ps(sum_low, sum_high);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                networkb0[p] = _mm_cvtss_f32(sum_low);
                /*for (; i < 784; ++i) {
                    networkb0[p] += network0[p][i] * tpool->TData0[TId][dtIndex][i];
                }*/

                networkb0[p] += network0[p][784];
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
                for (; i <= 128 - 64; i += 64) {
                    _mm_prefetch((const char*)(network1[p] + i + 64), _MM_HINT_T0);
                    _mm_prefetch((const char*)(networkn0 + i + 64), _MM_HINT_T0);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network1[p] + i), _mm256_loadu_ps(networkn0 + i), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network1[p] + i + 8), _mm256_loadu_ps(networkn0 + i + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network1[p] + i + 16), _mm256_loadu_ps(networkn0 + i + 16), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network1[p] + i + 24), _mm256_loadu_ps(networkn0 + i + 24), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network1[p] + i + 32), _mm256_loadu_ps(networkn0 + i + 32), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network1[p] + i + 40), _mm256_loadu_ps(networkn0 + i + 40), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network1[p] + i + 48), _mm256_loadu_ps(networkn0 + i + 48), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network1[p] + i + 56), _mm256_loadu_ps(networkn0 + i + 56), sum);
                }
                sum_high = _mm256_extractf128_ps(sum, 1);
                sum_low = _mm256_extractf128_ps(sum, 0);
                sum_low = _mm_add_ps(sum_low, sum_high);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                networkb1[p] = _mm_cvtss_f32(sum_low);
                /*for (; i < 128; ++i) {
                    networkb1[p] += network1[p][i] * networkn0[i];
                }*/

                networkb1[p] += network1[p][32];
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
                for (; i <= 32 - 32; i += 32) {
                    _mm_prefetch((const char*)(network2[p] + i + 32), _MM_HINT_T0);
                    _mm_prefetch((const char*)(networkn1 + i + 32), _MM_HINT_T0);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network2[p] + i), _mm256_loadu_ps(networkn1 + i), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network2[p] + i + 8), _mm256_loadu_ps(networkn1 + i + 8), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network2[p] + i + 16), _mm256_loadu_ps(networkn1 + i + 16), sum);
                    sum = _mm256_fmadd_ps(_mm256_loadu_ps(network2[p] + i + 24), _mm256_loadu_ps(networkn1 + i + 24), sum);
                }
                sum_high = _mm256_extractf128_ps(sum, 1);
                sum_low = _mm256_extractf128_ps(sum, 0);
                sum_low = _mm_add_ps(sum_low, sum_high);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                sum_low = _mm_hadd_ps(sum_low, sum_low);
                networkb2[p] = _mm_cvtss_f32(sum_low);
                /*for (; i < 32; ++i) {
                    networkb2[p] += network2[p][i] * networkn1[i];
                }*/

                networkb2[p] += network2[p][10];
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
                MSError += ((tpool->TData1[TId][dtIndex][p] - networkn2[p]) * (tpool->TData1[TId][dtIndex][p] - networkn2[p]));
            }

            //Back Propagation
            //Output Layer - Hidden Layer 2
            for (p = 0; p < 10; p++) {
                networkg2_neuron[p] = -rate * (networkn2[p] - tpool->TData1[TId][dtIndex][p]);
                networkg2[p][32] += networkg2_neuron[p];
                i = 0;
                factor = _mm256_set1_ps(networkg2_neuron[p]);
                for (; i <= 32 - 32; i += 32) {
                    _mm_prefetch((const char*)(networkn1 + i + 32), _MM_HINT_T0);
                    _mm_prefetch((const char*)(networkg2[p] + i + 32), _MM_HINT_T0);
                    _mm256_storeu_ps(networkg2[p] + i, _mm256_fmadd_ps(factor, _mm256_loadu_ps(networkn1 + i), _mm256_loadu_ps(networkg2[p] + i)));
                    _mm256_storeu_ps(networkg2[p] + i + 8, _mm256_fmadd_ps(factor, _mm256_loadu_ps(networkn1 + i + 8), _mm256_loadu_ps(networkg2[p] + i + 8)));
                    _mm256_storeu_ps(networkg2[p] + i + 16, _mm256_fmadd_ps(factor, _mm256_loadu_ps(networkn1 + i + 16), _mm256_loadu_ps(networkg2[p] + i + 16)));
                    _mm256_storeu_ps(networkg2[p] + i + 24, _mm256_fmadd_ps(factor, _mm256_loadu_ps(networkn1 + i + 24), _mm256_loadu_ps(networkg2[p] + i + 24)));
                }
                /*for (; i < 32; ++i) {
                    networkg2[p][i] += networkg2_neuron[p] * networkn1[i];
                }*/
            }
            //Hidden Layer 2 - Hidden Layer 1
            for (p = 0; p < 32; p++) {
                networkg1_neuron[p] = 0;
                for (q = 0; q < 10; q++) {
                    networkg1_neuron[p] += networkg2_neuron[q] * network2[q][p];
                }
                if (networkb1[p] >= 0) {
                    networkg1[p][128] = networkg1_neuron[p];
                } else {
                    networkg1_neuron[p] *= exp(networkb1[p]);
                    networkg1[p][128] = networkg1_neuron[p];
                }
                i = 0;
                factor = _mm256_set1_ps(networkg1_neuron[p]);
                for (; i <= 128 - 64; i += 64) {
                    _mm_prefetch((const char*)(networkn0 + i + 64), _MM_HINT_T0);
                    _mm_prefetch((const char*)(networkg1[p] + i + 64), _MM_HINT_T0);
                    _mm256_storeu_ps(networkg1[p] + i, _mm256_fmadd_ps(factor, _mm256_loadu_ps(networkn0 + i), _mm256_loadu_ps(networkg1[p] + i)));
                    _mm256_storeu_ps(networkg1[p] + i + 8, _mm256_fmadd_ps(factor, _mm256_loadu_ps(networkn0 + i + 8), _mm256_loadu_ps(networkg1[p] + i + 8)));
                    _mm256_storeu_ps(networkg1[p] + i + 16, _mm256_fmadd_ps(factor, _mm256_loadu_ps(networkn0 + i + 16), _mm256_loadu_ps(networkg1[p] + i + 16)));
                    _mm256_storeu_ps(networkg1[p] + i + 24, _mm256_fmadd_ps(factor, _mm256_loadu_ps(networkn0 + i + 24), _mm256_loadu_ps(networkg1[p] + i + 24)));
                    _mm256_storeu_ps(networkg1[p] + i + 32, _mm256_fmadd_ps(factor, _mm256_loadu_ps(networkn0 + i + 32), _mm256_loadu_ps(networkg1[p] + i + 32)));
                    _mm256_storeu_ps(networkg1[p] + i + 40, _mm256_fmadd_ps(factor, _mm256_loadu_ps(networkn0 + i + 40), _mm256_loadu_ps(networkg1[p] + i + 40)));
                    _mm256_storeu_ps(networkg1[p] + i + 48, _mm256_fmadd_ps(factor, _mm256_loadu_ps(networkn0 + i + 48), _mm256_loadu_ps(networkg1[p] + i + 48)));
                    _mm256_storeu_ps(networkg1[p] + i + 56, _mm256_fmadd_ps(factor, _mm256_loadu_ps(networkn0 + i + 56), _mm256_loadu_ps(networkg1[p] + i + 56)));
                }
                /*for (; i < 128; ++i) {
                    networkg1[p][i] += networkg1_neuron[p] * networkn0[i];
                }*/
            }
            //Hidden Layer 1 - Hidden Layer 0
            for (p = 0; p < 128; p++) {
                networkg0_neuron[p] = 0;
                for (q = 0; q < 32; q++) {
                    networkg0_neuron[p] += networkg1_neuron[q] * network1[q][p];
                }
                if (networkb0[p] >= 0) {
                    networkg0[p][784] = networkg0_neuron[p];
                } else {
                    networkg0_neuron[p] *= exp(networkb0[p]);
                    networkg0[p][784] = networkg0_neuron[p];
                }
                i = 0;
                factor = _mm256_set1_ps(networkg0_neuron[p]);
                for (; i <= 784 - 64; i += 64) {
                    _mm_prefetch((const char*)(networkn0 + i + 64), _MM_HINT_T0);
                    _mm_prefetch((const char*)(tpool->TData0[TId][dtIndex].data() + i + 64), _MM_HINT_T0);
                    _mm256_storeu_ps(networkg0[p] + i, _mm256_fmadd_ps(factor, _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i), _mm256_loadu_ps(networkg0[p] + i)));
                    _mm256_storeu_ps(networkg0[p] + i + 8, _mm256_fmadd_ps(factor, _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 8), _mm256_loadu_ps(networkg0[p] + i + 8)));
                    _mm256_storeu_ps(networkg0[p] + i + 16, _mm256_fmadd_ps(factor, _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 16), _mm256_loadu_ps(networkg0[p] + i + 16)));
                    _mm256_storeu_ps(networkg0[p] + i + 24, _mm256_fmadd_ps(factor, _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 24), _mm256_loadu_ps(networkg0[p] + i + 24)));
                    _mm256_storeu_ps(networkg0[p] + i + 32, _mm256_fmadd_ps(factor, _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 32), _mm256_loadu_ps(networkg0[p] + i + 32)));
                    _mm256_storeu_ps(networkg0[p] + i + 40, _mm256_fmadd_ps(factor, _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 40), _mm256_loadu_ps(networkg0[p] + i + 40)));
                    _mm256_storeu_ps(networkg0[p] + i + 48, _mm256_fmadd_ps(factor, _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 48), _mm256_loadu_ps(networkg0[p] + i + 48)));
                    _mm256_storeu_ps(networkg0[p] + i + 56, _mm256_fmadd_ps(factor, _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 56), _mm256_loadu_ps(networkg0[p] + i + 56)));
                }
                for (; i <= 784 - 16; i += 16) {
                    _mm_prefetch((const char*)(networkn0 + i + 16), _MM_HINT_T0);
                    _mm_prefetch((const char*)(tpool->TData0[TId][dtIndex].data() + i + 16), _MM_HINT_T0);
                    _mm256_storeu_ps(networkg0[p] + i, _mm256_fmadd_ps(factor, _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i), _mm256_loadu_ps(networkg0[p] + i)));
                    _mm256_storeu_ps(networkg0[p] + i + 8, _mm256_fmadd_ps(factor, _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i + 8), _mm256_loadu_ps(networkg0[p] + i + 8)));
                }
                /*for (; i < 784; ++i) {
                    networkg0[p][i] += networkg0_neuron[p] * tpool->TData0[TId][dtIndex][i];
                }*/
            }
            //Clear Temp
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
        for (; !networkg2_todoList[0] || !networkg1_todoList[0] || !networkg1_todoList[1] || !networkg1_todoList[2] || !networkg1_todoList[3] || !networkg0_todoList[0] || !networkg0_todoList[1] || !networkg0_todoList[2] || !networkg0_todoList[3] || !networkg0_todoList[4] || !networkg0_todoList[5] || !networkg0_todoList[6] || !networkg0_todoList[7] || !networkg0_todoList[8] || !networkg0_todoList[9] || !networkg0_todoList[10] || !networkg0_todoList[11] || !networkg0_todoList[12] || !networkg0_todoList[13] || !networkg0_todoList[14] || !networkg0_todoList[15];) {//Update All Blocks
            //Find a Free Block to update
            if (!networkg2_todoList[0]) {
                if (networkgs2_mtx[mtx_index_2].try_lock()) {
                    for (p = 0; p < 10; p++) {
                        i = 0;
                        for (; i <= 33 - 32; i += 32) {
                            _mm_prefetch((const char*)(networkg2[p] + i + 32), _MM_HINT_T0);
                            _mm_prefetch((const char*)(networkgs2[p] + i + 32), _MM_HINT_T0);
                            _mm256_storeu_ps(networkgs2[p] + i, _mm256_add_ps(_mm256_loadu_ps(networkg2[p] + i), _mm256_loadu_ps(networkgs2[p] + i)));
                            _mm256_storeu_ps(networkgs2[p] + i + 8, _mm256_add_ps(_mm256_loadu_ps(networkg2[p] + i + 8), _mm256_loadu_ps(networkgs2[p] + i + 8)));
                            _mm256_storeu_ps(networkgs2[p] + i + 16, _mm256_add_ps(_mm256_loadu_ps(networkg2[p] + i + 16), _mm256_loadu_ps(networkgs2[p] + i + 16)));
                            _mm256_storeu_ps(networkgs2[p] + i + 24, _mm256_add_ps(_mm256_loadu_ps(networkg2[p] + i + 24), _mm256_loadu_ps(networkgs2[p] + i + 24)));
                        }
                        for (; i < 33; ++i) {
                            networkgs2[p][i] += networkg2[p][i];
                        }
                    }
                    networkgs2_mtx[mtx_index_2].unlock();
                    networkg2_todoList[mtx_index_2] = true;
                }
            }
            for (mtx_index_1 = 0; mtx_index_1 < 4; mtx_index_1++) {
                if (!networkg1_todoList[mtx_index_1]) {
                    if (networkgs1_mtx[mtx_index_1].try_lock()) {
                        for (p = 0; p < 8; p++) {
                            i = 0;
                            for (; i <= 129 - 64; i += 64) {
                                _mm_prefetch((const char*)(networkg1[mtx_index_1 * 8 + p] + i + 64), _MM_HINT_T0);
                                _mm_prefetch((const char*)(networkgs1[mtx_index_1 * 8 + p] + i + 64), _MM_HINT_T0);
                                _mm256_storeu_ps(networkgs1[mtx_index_1 * 8 + p] + i, _mm256_add_ps(_mm256_loadu_ps(networkg1[mtx_index_1 * 8 + p] + i), _mm256_loadu_ps(networkgs1[mtx_index_1 * 8 + p] + i)));
                                _mm256_storeu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 8, _mm256_add_ps(_mm256_loadu_ps(networkg1[mtx_index_1 * 8 + p] + i + 8), _mm256_loadu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 8)));
                                _mm256_storeu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 16, _mm256_add_ps(_mm256_loadu_ps(networkg1[mtx_index_1 * 8 + p] + i + 16), _mm256_loadu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 16)));
                                _mm256_storeu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 24, _mm256_add_ps(_mm256_loadu_ps(networkg1[mtx_index_1 * 8 + p] + i + 24), _mm256_loadu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 24)));
                                _mm256_storeu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 32, _mm256_add_ps(_mm256_loadu_ps(networkg1[mtx_index_1 * 8 + p] + i + 32), _mm256_loadu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 32)));
                                _mm256_storeu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 40, _mm256_add_ps(_mm256_loadu_ps(networkg1[mtx_index_1 * 8 + p] + i + 40), _mm256_loadu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 40)));
                                _mm256_storeu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 48, _mm256_add_ps(_mm256_loadu_ps(networkg1[mtx_index_1 * 8 + p] + i + 48), _mm256_loadu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 48)));
                                _mm256_storeu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 56, _mm256_add_ps(_mm256_loadu_ps(networkg1[mtx_index_1 * 8 + p] + i + 56), _mm256_loadu_ps(networkgs1[mtx_index_1 * 8 + p] + i + 56)));
                            }
                            for (; i < 129; ++i) {
                                networkgs1[mtx_index_1 * 8 + p][i] += networkg1[mtx_index_1 * 8 + p][i];
                            }
                        }
                        networkgs1_mtx[mtx_index_1].unlock();
                        networkg1_todoList[mtx_index_1] = true;
                    }
                }
            }
            for (mtx_index_0 = 0; mtx_index_0 < 16; mtx_index_0++) {
                if (!networkg0_todoList[mtx_index_0]) {
                    if (networkgs0_mtx[mtx_index_0].try_lock()) {
                        for (p = 0; p < 8; p++) {
                            i = 0;
                            for (; i <= 785 - 64; i += 64) {
                                _mm_prefetch((const char*)(networkg0[mtx_index_0 * 8 + p] + i + 64), _MM_HINT_T0);
                                _mm_prefetch((const char*)(networkgs0[mtx_index_0 * 8 + p] + i + 64), _MM_HINT_T0);
                                _mm256_storeu_ps(networkgs0[mtx_index_0 * 8 + p] + i, _mm256_add_ps(_mm256_loadu_ps(networkg0[mtx_index_0 * 8 + p] + i), _mm256_loadu_ps(networkgs0[mtx_index_0 * 8 + p] + i)));
                                _mm256_storeu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 8, _mm256_add_ps(_mm256_loadu_ps(networkg0[mtx_index_0 * 8 + p] + i + 8), _mm256_loadu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 8)));
                                _mm256_storeu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 16, _mm256_add_ps(_mm256_loadu_ps(networkg0[mtx_index_0 * 8 + p] + i + 16), _mm256_loadu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 16)));
                                _mm256_storeu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 24, _mm256_add_ps(_mm256_loadu_ps(networkg0[mtx_index_0 * 8 + p] + i + 24), _mm256_loadu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 24)));
                                _mm256_storeu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 32, _mm256_add_ps(_mm256_loadu_ps(networkg0[mtx_index_0 * 8 + p] + i + 32), _mm256_loadu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 32)));
                                _mm256_storeu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 40, _mm256_add_ps(_mm256_loadu_ps(networkg0[mtx_index_0 * 8 + p] + i + 40), _mm256_loadu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 40)));
                                _mm256_storeu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 48, _mm256_add_ps(_mm256_loadu_ps(networkg0[mtx_index_0 * 8 + p] + i + 48), _mm256_loadu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 48)));
                                _mm256_storeu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 56, _mm256_add_ps(_mm256_loadu_ps(networkg0[mtx_index_0 * 8 + p] + i + 56), _mm256_loadu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 56)));
                            }
                            for (; i <= 785 - 16; i += 16) {
                                _mm_prefetch((const char*)(networkg0[mtx_index_0 * 8 + p] + i + 16), _MM_HINT_T0);
                                _mm_prefetch((const char*)(networkgs0[mtx_index_0 * 8 + p] + i + 16), _MM_HINT_T0);
                                _mm256_storeu_ps(networkgs0[mtx_index_0 * 8 + p] + i, _mm256_add_ps(_mm256_loadu_ps(networkg0[mtx_index_0 * 8 + p] + i), _mm256_loadu_ps(networkgs0[mtx_index_0 * 8 + p] + i)));
                                _mm256_storeu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 8, _mm256_add_ps(_mm256_loadu_ps(networkg0[mtx_index_0 * 8 + p] + i + 8), _mm256_loadu_ps(networkgs0[mtx_index_0 * 8 + p] + i + 8)));
                            }
                            for (; i < 785; ++i) {
                                networkgs0[mtx_index_0 * 8 + p][i] += networkg0[mtx_index_0 * 8 + p][i];
                            }
                        }
                        networkgs0_mtx[mtx_index_0].unlock();
                        networkg0_todoList[mtx_index_0] = true;
                    }
                }
            }
        }
        //Init todoList
        networkg2_todoList[0] = false;
        for (i = 0; i < 4; i++) {
            networkg1_todoList[i] = false;
        }
        for (i = 0; i < 16; i++) {
            networkg0_todoList[i] = false;
        }

        //Clear Temp
        fill(networkg0[0], networkg0[0] + 100480, 0);
        fill(networkg1[0], networkg1[0] + 4128, 0);
        fill(networkg2[0], networkg2[0] + 310, 0);

        mtx.lock();
        MSETotal += MSError;//Add Lost to Global Cost of this Batch
        reportI++;//This Thread Finished BP
        if (reportI == tpool->TSize) {//If All Threads Finished BP
            //Update Weights & Bias
            memcpy(network0, networkgs0, 401920);
            memcpy(network1, networkgs1, 16512);
            memcpy(network2, networkgs2, 1240);
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


inline void TP::init(int size) {
    TSize = size;
    for (int i = 0; i < size; i++) {
        TGate.emplace_back(false);
        TFree.emplace_back(true);
        TData0.push_back({});
        TData1.push_back({});
        thread(trainNet, i).detach();
    }
}


void train(float rate, float aim) {
    std::cout << "Gradient loss function: Cross Entropy" << std::endl << "------------------------------" << std::endl;
    int i{}, c, w, dti;
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
                    for (w = 0; w < batchSize; w += batchSize / tpool->TSize) {
                        for (dti = 0; dti < batchSize / tpool->TSize; dti++) {
                            temp0[dti] = train_image[batchSize * c + w + dti];
                            temp1[dti] = train_label[batchSize * c + w + dti];
                        }
                        tpool->add(temp0, temp1);
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
        for (int j = 0; j < 785; ++j) {
            network0[i][j] = dis(gen);
        }
    }
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 129; ++j) {
            network1[i][j] = dis(gen);
        }
    }
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 33; ++j) {
            network2[i][j] = dis(gen);
        }
    }
    memcpy(networkgs0, network0, 401920);
    memcpy(networkgs1, network1, 16512);
    memcpy(networkgs2, network2, 1240);

    batchSize = 50;
    tpool->init(10);//BatchSize must be a positive integer multiple of the number of threads

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
    read_Mnist_Label(train_label_name, train_labels);
    std::cout << "3/3 Convert Training Data" << endl;
    convert_array_label(train_labels);

    std::cout << "Ready..." << std::endl;
    train(rate, aim);
}