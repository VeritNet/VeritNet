/*
* g++ -O3 -march=native -finline-functions -funroll-loops -mavx2 -o v3.exe v3.cpp
* 如果需要极致的速度，请将O3替换为Ofast，但这将导致部分计算中出现精度丢失
* If extreme speed is required, please replace O3 with Ofast, but this will result in loss of accuracy in some calculations
* Test Version 2024.7.22.3
*/

#include <iostream>
#include <cstring>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <thread>
#include <mutex>
#include <deque>
#include <chrono>
#include <atomic>
#include <immintrin.h>

#include "loadMNIST.h"


using namespace std;


std::mutex mtx;
class TP {
public:
    deque<atomic<bool>> TGate;
    deque<atomic<bool>> TFree;
    vector< vector<vector<float>> > TData0;
    vector< vector<vector<float>> > TData1;
    int TSize;
    void init(int size);
    inline void add(vector<vector<float>>& inputDt0, vector<vector<float>>& inputDt1) {
        bool noBreak = true;
        while (noBreak) {
            for (int i = 0; i < TSize; i++) {
                if (TFree[i].load()) {
                    TFree[i].store(false);
                    mtx.lock();
                    TData0[i] = inputDt0;
                    TData1[i] = inputDt1;
                    mtx.unlock();
                    TGate[i].store(true);
                    noBreak = false;
                    break;
                }
            }
        }
    }
};
TP* tpool = new TP;
int dts;
float network0[128][784 + 1];
float network1[30][128 + 1];
float network2[10][30 + 1];
int batchSize;
float MSETotal;
float networkgs0[128][784 + 1];
std::vector<std::mutex> networkgs0_mtx(16);
float networkgs1[30][128 + 1];
std::vector<std::mutex> networkgs1_mtx(3);
float networkgs2[10][30 + 1];
std::vector<std::mutex> networkgs2_mtx(1);
bool gate;
int reportI = 0;
float rate, aim, err;
inline void trainNet(int TId) {
    float networkg0[128][784 + 1] = { 0 };//线程累加梯度
    vector<bool> networkg0_todoList(16);
    float networkg1[30][128 + 1] = { 0 };
    vector<bool> networkg1_todoList(3);
    float networkg2[10][30 + 1] = { 0 };
    vector<bool> networkg2_todoList(1);
    float MSError;

    float SSum;
    int i, p, q, dtIndex;
    int mtx_index_0, mtx_index_1, mtx_index_2;
    __m256 sum;

    float networkn0[128] = { 0 };
    float networkn1[30] = { 0 };
    float networkn2[10] = { 0 };
    float networkb0[128] = { 0 };
    float networkb1[30] = { 0 };
    float networkb2[10] = { 0 };

    //神经元的梯度等于神经元偏置的梯度
    float networkg0_neuron[128] = { 0 };
    float networkg1_neuron[30] = { 0 };
    float networkg2_neuron[10] = { 0 };


    for (;;) {
        for (; tpool->TGate[TId].load() == false;) {}
        tpool->TGate[TId].store(false);
        for (dtIndex = tpool->TData0[TId].size() - 1; dtIndex >= 0; dtIndex--) {
            for (p = 0; p < 128; p++) {
                sum = _mm256_setzero_ps();
                for (i = 0; i < 784 - (784 % 8); i += 8) {
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(network0[p] + i), _mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i)));
                }
                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(sum, sum, 1));
                sum = _mm256_hadd_ps(sum, sum);
                sum = _mm256_hadd_ps(sum, sum);
                networkb0[p] = _mm_cvtss_f32(_mm256_castps256_ps128(sum));
                for (; i < 784; ++i) {
                    networkb0[p] += network0[p][i] * tpool->TData0[TId][dtIndex][i];
                }

                networkb0[p] += network0[p][784];
                if (networkb0[p] >= 0) {
                    networkn0[p] = networkb0[p];
                } else {
                    networkn0[p] = exp(networkb0[p]) - 1;
                }
            }
            for (p = 0; p < 30; p++) {
                sum = _mm256_setzero_ps();
                for (i = 0; i < 128 - (128 % 8); i += 8) {
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(network1[p] + i), _mm256_loadu_ps(networkn0 + i)));
                }
                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(sum, sum, 1));
                sum = _mm256_hadd_ps(sum, sum);
                sum = _mm256_hadd_ps(sum, sum);
                networkb1[p] = _mm_cvtss_f32(_mm256_castps256_ps128(sum));
                for (; i < 128; ++i) {
                    networkb1[p] += network1[p][i] * networkn0[i];
                }

                networkb1[p] += network1[p][30];
                if (networkb1[p] >= 0) {
                    networkn1[p] = networkb1[p];
                } else {
                    networkn1[p] = exp(networkb1[p]) - 1;
                }
            }
            for (p = 0; p < 10; p++) {
                sum = _mm256_setzero_ps();
                for (i = 0; i < 30 - (30 % 8); i += 8) {
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(network2[p] + i), _mm256_loadu_ps(networkn1 + i)));
                }
                sum = _mm256_add_ps(sum, _mm256_permute2f128_ps(sum, sum, 1));
                sum = _mm256_hadd_ps(sum, sum);
                sum = _mm256_hadd_ps(sum, sum);
                networkb2[p] = _mm_cvtss_f32(_mm256_castps256_ps128(sum));
                for (; i < 30; ++i) {
                    networkb2[p] += network2[p][i] * networkn1[i];
                }

                networkb2[p] += network2[p][10];
            }
            SSum = 0;
            for (i = 0; i < 10; i++) {
                SSum += exp(networkb2[i]);
            }
            for (i = 0; i < 10; i++) {
                networkn2[i] += exp(networkb2[i]) / SSum;
            }

            MSError = 0;
            for (p = 0; p < 10; p++) {
                MSError += ((tpool->TData1[TId][dtIndex][p] - networkn2[p]) * (tpool->TData1[TId][dtIndex][p] - networkn2[p]));
            }

            for (p = 0; p < 10; p++) {
                networkg2_neuron[p] = -rate * (networkn2[p] - tpool->TData1[TId][dtIndex][p]);
                networkg2[p][30] += networkg2_neuron[p];
                for (i = 0; i <= 30 - 8; i += 8) {
                    _mm256_storeu_ps(networkg2[p] + i, _mm256_add_ps(  _mm256_mul_ps(_mm256_loadu_ps(networkn1 + i), _mm256_set1_ps(networkg2_neuron[p])), _mm256_loadu_ps(networkg2[p] + i)  ));
                }
                for (; i < 30; ++i) {
                    networkg2[p][i] += networkg2_neuron[p] * networkn1[i];
                }
            }
            for (p = 0; p < 30; p++) {
                ////////计算本层每个神经元的梯度
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
                ////////
                for (i = 0; i <= 128 - 8; i += 8) {
                    _mm256_storeu_ps(networkg1[p] + i, _mm256_add_ps(  _mm256_mul_ps(_mm256_loadu_ps(networkn0 + i), _mm256_set1_ps(networkg1_neuron[p])), _mm256_loadu_ps(networkg1[p] + i)  ));
                }
                for (; i < 128; ++i) {
                    networkg1[p][i] += networkg1_neuron[p] * networkn0[i];
                }
            }
            for (p = 0; p < 128; p++) {
                networkg0_neuron[p] = 0;
                for (q = 0; q < 30; q++) {
                    networkg0_neuron[p] += networkg1_neuron[q] * network1[q][p];
                }
                if (networkb0[p] >= 0) {
                    networkg0[p][784] = networkg0_neuron[p];
                } else {
                    networkg0_neuron[p] *= exp(networkb0[p]);
                    networkg0[p][784] = networkg0_neuron[p];
                }
                for (i = 0; i <= 784 - 8; i += 8) {
                    _mm256_storeu_ps(networkg0[p] + i, _mm256_add_ps(  _mm256_mul_ps(_mm256_loadu_ps(tpool->TData0[TId][dtIndex].data() + i), _mm256_set1_ps(networkg0_neuron[p])), _mm256_loadu_ps(networkg0[p] + i)));
                }
                for (; i < 784; ++i) {
                    networkg0[p][i] += networkg0_neuron[p] * tpool->TData0[TId][dtIndex][i];
                }
            }
            std::fill(networkn0, networkn0 + 128, 0);
            std::fill(networkn1, networkn1 + 30, 0);
            std::fill(networkn2, networkn2 + 10, 0);
            std::fill(networkb0, networkb0 + 128, 0);
            std::fill(networkb1, networkb1 + 30, 0);
            std::fill(networkb2, networkb2 + 10, 0);
        }

        //MiniBatch SGD
        networkgs2_mtx[0].lock();
        for (p = 0; p < 10; p++) {
            for (i = 0; i + 7 < 16; i += 8) {
                _mm256_storeu_ps(networkgs2[p] + i, _mm256_add_ps(_mm256_loadu_ps(networkg2[p] + i), _mm256_loadu_ps(networkgs2[p] + i)));
            }
            for (; i < 16; ++i) {
                networkgs2[p][i] += networkg2[p][i];
            }
        }
        networkgs2_mtx[0].unlock();

        mtx_index_1 = 0;
        mtx_index_0 = 0;
        for (; !networkg1_todoList[0] || !networkg1_todoList[1] || !networkg1_todoList[2] || !networkg0_todoList[0] || !networkg0_todoList[1] || !networkg0_todoList[2] || !networkg0_todoList[3] || !networkg0_todoList[4] || !networkg0_todoList[5] || !networkg0_todoList[6] || !networkg0_todoList[7] || !networkg0_todoList[8] || !networkg0_todoList[9] || !networkg0_todoList[10] || !networkg0_todoList[11] || !networkg0_todoList[12] || !networkg0_todoList[13] || !networkg0_todoList[14] || !networkg0_todoList[15];) {
            for (mtx_index_1 = 0; mtx_index_1 < 3; mtx_index_1++) {
                if (!networkg1_todoList[mtx_index_1]) {
                    if (networkgs1_mtx[mtx_index_1].try_lock()) {
                        for (p = 0; p < 10; p++) {
                            for (i = 0; i + 7 < 129; i += 8) {
                                _mm256_storeu_ps(networkgs1[mtx_index_1 * 10 + p] + i, _mm256_add_ps(_mm256_loadu_ps(networkg1[mtx_index_1 * 10 + p] + i), _mm256_loadu_ps(networkgs1[mtx_index_1 * 10 + p] + i)));
                            }
                            for (; i < 129; ++i) {
                                networkgs1[mtx_index_1 * 10 + p][i] += networkg1[mtx_index_1 * 10 + p][i];
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
                            for (i = 0; i + 7 < 785; i += 8) {
                                _mm256_storeu_ps(networkgs0[mtx_index_0 * 8 + p] + i, _mm256_add_ps(_mm256_loadu_ps(networkg0[mtx_index_0 * 8 + p] + i), _mm256_loadu_ps(networkgs0[mtx_index_0 * 8 + p] + i)));
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
        for (i = 0; i < 3; i++) {
            networkg1_todoList[i] = false;
        }
        for (i = 0; i < 16; i++) {
            networkg0_todoList[i] = false;
        }

        fill(networkg0[0], networkg0[0] + 100480, 0);
        fill(networkg1[0], networkg1[0] + 3870, 0);
        fill(networkg2[0], networkg2[0] + 310, 0);

        mtx.lock();
        MSETotal += MSError;
        reportI++;
        if (reportI == batchSize / tpool->TSize) {
            memcpy(network0, networkgs0, 401920);
            memcpy(network1, networkgs1, 15480);
            memcpy(network2, networkgs2, 1240);
            err += MSETotal;
            MSETotal = 0;
            reportI = 0;
            gate = true;
        }
        mtx.unlock();
        tpool->TFree[TId].store(true);
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
    std::cout << "Gradient loss function: Cross Entropy" << std::endl;
    int i = 0;
    vector<vector<float>> temp0(batchSize / tpool->TSize);
    vector<vector<float>> temp1(batchSize / tpool->TSize);
    while (true) {
        i++;
        err = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int c = 0; c < dts / batchSize; c++) {
            while (true) {
                if (gate == true) {
                    gate = false;
                    for (int w = 0; w < batchSize; w += batchSize / tpool->TSize) {
                        for (int dti = 0; dti < batchSize / tpool->TSize; dti++) {
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
        std::chrono::duration<double> duration = end - start;
        std::cout << "spent time: " << duration.count() << "s" << endl;

        rate *= 1.1f;

        if (err <= aim) {
            std::cout << "Loss: " << err << std::endl;
            std::cout << ">>> finished " << dts * i << " steps (" << i << " rounds) gradient descent" << std::endl;
            break;
        } else {
            std::cout << "Epoch: " << i << "  Training: " << dts * i << "  MSE: " << err << "(Average: " << err / dts << ")" << " rate: " << rate << std::endl;
        }
    }
}



int main() {
    std::cout << "1/3 Init Data" << endl;
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
    for (int i = 0; i < 30; ++i) {
        for (int j = 0; j < 129; ++j) {
            network1[i][j] = dis(gen);
        }
    }
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 31; ++j) {
            network2[i][j] = dis(gen);
        }
    }
    memcpy(networkgs0, network0, 401920);
    memcpy(networkgs1, network1, 15480);
    memcpy(networkgs2, network2, 1240);

    batchSize = 40;
    tpool->init(10);//BatchSize must be a positive integer multiple of the number of threads. BatchSize必须是线程数的正整数倍

    //模型精度是Float，4个字节
    rate = 0.003;//Learning Rate 学习率
    aim = 10;//Aiming Loss(MSE in total) 目标损失值（是一个Epoch全部数据的MSE的总和）
    dts = 60000;//Data Limited(Max 60000) 多少个训练数据（最大60000）

    std::cout << "2/3 Load Training Data" << endl;

    //====================================================================
    //转自 https://www.cnblogs.com/ppDoo/p/13261258.html                //
    //仅用于测试                                                        //
    char train_image_name[] = "train-images.idx3-ubyte";            //
    char train_label_name[] = "train-labels.idx1-ubyte";            //
    vector< vector<int> > train_feature_vector;                     //
    read_Mnist_Images(train_image_name, train_feature_vector, dts); //
    convert_array_image(train_feature_vector);                      //
    vector<int> train_labels;                                       //
    read_Mnist_Label(train_label_name, train_labels);               //
    convert_array_label(train_labels);                              //
    //====================================================================

    std::cout << "3/3 Ready" << endl;
    system("cls");
    train(rate, aim);
}
