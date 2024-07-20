/*
* 编译命令：g++ -O3 -march=native -finline-functions -funroll-loops -mavx2 -o dnn_nnTest.exe dnn_nnTest.cpp
* 尽量Windows用Mingw，Linux用clang，用上述命令编译，否则性能会变差（尤其不要用MSVC）
* 请将MNIST数据集的两个训练数据文件（解压后的）放到同级目录下，或修改main函数里面的两个文件路径
* 根据大多数人电脑配置，我选了AVX2指令集，如果嫌旧可以替换成AVX512，可以去Intel官网查，Linux也可以打印自己CPU的信息
* 本文件代码是由VeritNet引擎根据蓝图自动生成的，所以几乎无法手动维护（可读性较差），本来变量名都是用节点Id生成的，后来手动替换了一下几个关键变量名
* main函数有注释的部分，自行调整学习率、Batch Size以及Data Size，详见该处
* ----------------------------------------
* 神经元个数    128     30      10
* 激活函数      Elu     Elu     Softmax
* ----------------------------------------
* 目前梯度爆炸严重，以前很少跑过完整MNIST，用PyTorch发现也会爆炸，正在思考有没有不加优化器防止爆炸的方法
* 目前还有几个地方没有优化，把上述问题解决后再继续优化代码
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include <thread>
#include <mutex>
#include <deque>
#include <chrono>
#include <atomic>
#include <windows.h>
#include <immintrin.h>

#include "loadMNIST.h"


using namespace std;


inline void dot_product(size_t n, const float* a, const float* b, float& re) {
    __m256 sum = _mm256_setzero_ps(); // 初始化为0
    size_t i;
    for (i = 0; i < n - (n % 8); i += 8) {
        __m256 ai = _mm256_loadu_ps(a + i); // 加载a[i]到a[i+7]
        __m256 bi = _mm256_loadu_ps(b + i); // 加载b[i]到b[i+7]
        __m256 prod = _mm256_mul_ps(ai, bi); // 计算a[i]*b[i]到a[i+7]*b[i+7]
        sum = _mm256_add_ps(sum, prod); // 累加结果
    }
    // 水平加法，将8个float累加到前两个元素
    sum = _mm256_hadd_ps(sum, sum);
    sum = _mm256_hadd_ps(sum, sum);
    re = _mm_cvtss_f32(_mm256_castps256_ps128(sum));
    // 处理剩余的元素
    for (; i < n; ++i) {
        re += a[i] * b[i];
    }
}
inline void hadamard_product(float* a, float* b, float* result, int size) {
    int i = 0;
    for (; i < size - (size % 8); i += 8) {
        __m256 vector1 = _mm256_loadu_ps(a + i); // 加载对齐的单精度浮点数
        __m256 vector2 = _mm256_loadu_ps(b + i); // 加载对齐的单精度浮点数
        __m256 res = _mm256_mul_ps(vector1, vector2); // 计算Hadamard乘积
        _mm256_storeu_ps(result + i, res); // 存储结果
    }
    // 处理剩余的元素
    for (; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}
inline void scale_product(int size, float* arr, float scalar) {
    __m256 factor = _mm256_set1_ps(scalar); // 将标量设置为向量的每个部分
    int i;
    for (i = 0; i <= size - 8; i += 8) {
        __m256 vec = _mm256_loadu_ps(&arr[i]); // 加载arr的8个元素
        __m256 result = _mm256_mul_ps(vec, factor); // 将它们乘以标量
        _mm256_storeu_ps(&arr[i], result); // 将结果存储回arr
    }
    // 处理size不是8的倍数的情况
    for (; i < size; ++i) {
        arr[i] *= scalar;
    }
}
inline void add_arrays(int n, float* a, float* b) {
    int i;
    // 处理可以被8整除的部分
    for (i = 0; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vb = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(b + i, vb);
    }
    // 处理剩余的元素
    for (; i < n; ++i) {
        b[i] += a[i];
    }
}
std::mutex mtx;
class TP {
public:
    deque<atomic<bool>> TGate;
    deque<atomic<bool>> TFree;
    vector<vector<float>> TData0;
    vector<vector<float>> TData1;
    int TSize;
    void init(int size);
    inline void add(vector<float>* inputDt0, vector<float>* inputDt1) {
        bool noBreak = true;
        while (noBreak) {
            for (int i = 0; i < TSize; i++) {
                if (TFree[i].load()) {
                    TFree[i].store(false);
                    mtx.lock();
                    TData0[i] = *inputDt0;
                    TData1[i] = *inputDt1;
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
float networkn0[128] = { 0 };
float networkn1[30] = { 0 };
float networkn2[10] = { 0 };
float networkb0[128] = { 0 };
float networkb1[30] = { 0 };
float networkb2[10] = { 0 };
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
    float networkg0[128][784 + 1];
    memcpy(networkg0, network0, 401920);
    vector<bool> networkg0_todoList(16);
    float networkg1[30][128 + 1];
    memcpy(networkg1, network1, 15480);
    vector<bool> networkg1_todoList(3);
    float networkg2[10][30 + 1];
    memcpy(networkg2, network2, 1240);
    vector<bool> networkg2_todoList(1);
    float MSError;
    std::vector<float> rMEdN(10);

    int m, n;
    float SSum;
    int i, l, p;
    float averagenN;
    int s;
    int mtx_index_0;
    int mtx_index_1;
    int mtx_index_2;

    float networkn0[128] = { 0 };
    float networkn1[30] = { 0 };
    float networkn2[10] = { 0 };
    float networkb0[128] = { 0 };
    float networkb1[30] = { 0 };
    float networkb2[10] = { 0 };


    for (;;) {
        for (; tpool->TGate[TId].load() == false;) {}
        tpool->TGate[TId].store(false);
        for (m = 0; m < 128; m++) {
            dot_product(784, network0[m], tpool->TData0[TId].data(), networkb0[m]);
            networkb0[m] += network0[m][784];
            if (networkb0[m] >= 0) {
                networkn0[m] = networkb0[m];
            } else {
                networkn0[m] = exp(networkb0[m]) - 1;
            }
        }
        for (m = 0; m < 30; m++) {
            dot_product(128, network1[m], networkn0, networkb1[m]);
            networkb1[m] += network1[m][30];
            if (networkb1[m] >= 0) {
                networkn1[m] = networkb1[m];
            } else {
                networkn1[m] = exp(networkb1[m]) - 1;
            }
        }
        for (n = 0; n < 10; n++) {
            dot_product(30, network2[n], networkn1, networkb2[n]);
            networkb2[n] += network2[n][10];
        }
        SSum = 0;
        for (i = 0; i < 10; i++) {
            SSum += exp(networkb2[i]);
        }
        for (i = 0; i < 10; i++) {
            networkn2[i] += exp(networkb2[i]) / SSum;
        }

        MSError = 0;
        for (l = 0; l < 10; l++) {
            MSError += ((tpool->TData1[TId][l] - networkn2[l]) * (tpool->TData1[TId][l] - networkn2[l]));
        }
        for (l = 0; l < 10; l++) {
            rMEdN[l] = -rate * (networkn2[l] - tpool->TData1[TId][l]);
        }

        for (p = 0; p < 10; p++) {
            memcpy(networkg2[p], networkn1, 120);
            scale_product(30, networkg2[p], rMEdN[p]);
            networkg2[p][30] = rMEdN[p];
        }
        for (p = 0; p < 30; p++) {
            averagenN = 0;
            for (s = 0; s < 10; s++) {
                averagenN += rMEdN[s] * network2[s][p];
            }
            if (networkb1[p] >= 0) {
                networkg1[p][128] = averagenN;
            } else {
                networkg1[p][128] = averagenN * exp(networkb1[p]);
            }
            memcpy(networkg1[p], networkn0, 512);
            scale_product(128, networkg1[p], networkg1[p][128]);
        }
        for (p = 0; p < 128; p++) {
            averagenN = 0;
            for (s = 0; s < 30; s++) {
                averagenN += networkg1[s][128] * network1[s][p];
            }
            if (networkb0[p] >= 0) {
                networkg0[p][784] = averagenN;
            } else {
                networkg0[p][784] = averagenN * exp(networkb0[p]);
            }
            std::copy(tpool->TData0[TId].begin(), tpool->TData0[TId].end(), std::begin(networkg0[p]));
            scale_product(784, networkg0[p], networkg0[p][784]);
        }
        std::fill(networkn0, networkn0 + 128, 0);
        std::fill(networkn1, networkn1 + 30, 0);
        std::fill(networkn2, networkn2 + 10, 0);
        std::fill(networkb0, networkb0 + 128, 0);
        std::fill(networkb1, networkb1 + 30, 0);
        std::fill(networkb2, networkb2 + 10, 0);
        networkgs2_mtx[0].lock();
        for (p = 0; p < 10; p++) {
            add_arrays(16, networkg2[p], networkgs2[p]);
        }
        networkgs2_mtx[0].unlock();

        mtx_index_1 = 0;
        mtx_index_0 = 0;
        for (; !networkg1_todoList[0] || !networkg1_todoList[1] || !networkg1_todoList[2] || !networkg0_todoList[0] || !networkg0_todoList[1] || !networkg0_todoList[2] || !networkg0_todoList[3] || !networkg0_todoList[4] || !networkg0_todoList[5] || !networkg0_todoList[6] || !networkg0_todoList[7] || !networkg0_todoList[8] || !networkg0_todoList[9] || !networkg0_todoList[10] || !networkg0_todoList[11] || !networkg0_todoList[12] || !networkg0_todoList[13] || !networkg0_todoList[14] || !networkg0_todoList[15];) {
            for (mtx_index_1 = 0; mtx_index_1 < 3; mtx_index_1++) {
                if (!networkg1_todoList[mtx_index_1]) {
                    if (networkgs1_mtx[mtx_index_1].try_lock()) {
                        add_arrays(129, networkg1[mtx_index_1 * 10], networkgs1[mtx_index_1 * 10]);
                        add_arrays(129, networkg1[mtx_index_1 * 10 + 1], networkgs1[mtx_index_1 * 10 + 1]);
                        add_arrays(129, networkg1[mtx_index_1 * 10 + 2], networkgs1[mtx_index_1 * 10 + 2]);
                        add_arrays(129, networkg1[mtx_index_1 * 10 + 3], networkgs1[mtx_index_1 * 10 + 3]);
                        add_arrays(129, networkg1[mtx_index_1 * 10 + 4], networkgs1[mtx_index_1 * 10 + 4]);
                        add_arrays(129, networkg1[mtx_index_1 * 10 + 5], networkgs1[mtx_index_1 * 10 + 5]);
                        add_arrays(129, networkg1[mtx_index_1 * 10 + 6], networkgs1[mtx_index_1 * 10 + 6]);
                        add_arrays(129, networkg1[mtx_index_1 * 10 + 7], networkgs1[mtx_index_1 * 10 + 7]);
                        add_arrays(129, networkg1[mtx_index_1 * 10 + 8], networkgs1[mtx_index_1 * 10 + 8]);
                        add_arrays(129, networkg1[mtx_index_1 * 10 + 9], networkgs1[mtx_index_1 * 10 + 9]);
                        networkgs1_mtx[mtx_index_1].unlock();
                        networkg1_todoList[mtx_index_1] = true;
                    }
                }
            }
            for (mtx_index_0 = 0; mtx_index_0 < 16; mtx_index_0++) {
                if (!networkg0_todoList[mtx_index_0]) {
                    if (networkgs0_mtx[mtx_index_0].try_lock()) {
                        add_arrays(785, networkg0[mtx_index_0 * 8], networkgs0[mtx_index_0 * 8]);
                        add_arrays(785, networkg0[mtx_index_0 * 8 + 1], networkgs0[mtx_index_0 * 8 + 1]);
                        add_arrays(785, networkg0[mtx_index_0 * 8 + 2], networkgs0[mtx_index_0 * 8 + 2]);
                        add_arrays(785, networkg0[mtx_index_0 * 8 + 3], networkgs0[mtx_index_0 * 8 + 3]);
                        add_arrays(785, networkg0[mtx_index_0 * 8 + 4], networkgs0[mtx_index_0 * 8 + 4]);
                        add_arrays(785, networkg0[mtx_index_0 * 8 + 5], networkgs0[mtx_index_0 * 8 + 5]);
                        add_arrays(785, networkg0[mtx_index_0 * 8 + 6], networkgs0[mtx_index_0 * 8 + 6]);
                        add_arrays(785, networkg0[mtx_index_0 * 8 + 7], networkgs0[mtx_index_0 * 8 + 7]);
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

        mtx.lock();
        MSETotal += MSError;
        reportI++;
        if (reportI == batchSize) {
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
    while (true) {
        i++;
        err = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int c = 0; c < dts / batchSize; c++) {
            while (true) {
                if (gate == true) {
                    gate = false;
                    for (int w = 0; w < batchSize; w++) {
                        tpool->add(&train_image[batchSize * c + w], &train_label[batchSize * c + w]);
                    }
                    break;
                }
            }
        }
        while (gate == false);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "spent time: " << duration.count() << "s" << endl;

        if (i % 5 == 0) {
            rate *= 1.1f;
        }
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
    batchSize = 20;
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
    tpool->init(6);

    //模型精度是Float，4个字节
    rate = 0.00001;//学习率
    aim = 10;//目标损失值（是一个Epoch全部数据的MSE的总和）
    dts = 60000;//多少个训练数据（最大60000）

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