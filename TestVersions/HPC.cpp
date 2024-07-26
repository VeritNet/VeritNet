#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <chrono>
#include <immintrin.h>

float dot_product(const float* a, const float* b, int n) {
  __m256 sum1, sum2, sum3, sum4;
  __m128 sum_high, sum_low;

  float re = 0;
  int i = 0;

  // 使用 AVX2 处理 16 个元素的块, 并展开循环
  for (; i <= n - 64; i += 64) {
    // 软件预取，提高缓存命中率
    _mm_prefetch((const char*)(a + i + 64), _MM_HINT_T0);
    _mm_prefetch((const char*)(b + i + 64), _MM_HINT_T0);

    // 使用 vfmadd231ps 指令 (如果可用) 来提高性能
    #ifdef __FMA__
      sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), _mm256_setzero_ps());
      sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
      sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), sum2);
      sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), sum3);
      sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 32), _mm256_loadu_ps(b + i + 32), sum4);
      sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 40), _mm256_loadu_ps(b + i + 40), sum1);
      sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 48), _mm256_loadu_ps(b + i + 48), sum2);
      sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 56), _mm256_loadu_ps(b + i + 56), sum3);
    #else
      sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), _mm256_setzero_ps());
      sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), _mm256_setzero_ps());
      sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), _mm256_setzero_ps());
      sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), _mm256_setzero_ps());
      sum1 = _mm256_add_ps(sum1, sum2);
      sum3 = _mm256_add_ps(sum3, sum4);
      sum1 = _mm256_add_ps(sum1, sum3);
      sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 32), _mm256_loadu_ps(b + i + 32), _mm256_setzero_ps());
      sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 40), _mm256_loadu_ps(b + i + 40), _mm256_setzero_ps());
      sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 48), _mm256_loadu_ps(b + i + 48), _mm256_setzero_ps());
      sum2 = _mm256_add_ps(sum2, sum3);
      sum4 = _mm256_add_ps(sum4, _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 56), _mm256_loadu_ps(b + i + 56), _mm256_setzero_ps()));
      sum1 = _mm256_add_ps(sum1, sum2);
      sum1 = _mm256_add_ps(sum1, sum4);
    #endif
    

    sum_high = _mm256_extractf128_ps(sum1, 1);
    sum_low = _mm256_extractf128_ps(sum1, 0);
    sum_low = _mm_add_ps(sum_low, sum_high);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    re += _mm_cvtss_f32(sum_low);
  }

  for (; i <= n - 32; i += 32) {
    _mm_prefetch((const char*)(a + i + 64), _MM_HINT_T0);
    _mm_prefetch((const char*)(b + i + 64), _MM_HINT_T0);

    #ifdef __FMA__
      sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), _mm256_setzero_ps());
      sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
      sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), sum2);
      sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), sum3);
    #else
      sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), _mm256_setzero_ps());
      sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), _mm256_setzero_ps());
      sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), _mm256_setzero_ps());
      sum4 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), _mm256_setzero_ps());
      sum1 = _mm256_add_ps(sum1, sum2);
      sum3 = _mm256_add_ps(sum3, sum4);
      sum1 = _mm256_add_ps(sum1, sum3);
    #endif

    sum_high = _mm256_extractf128_ps(sum1, 1);
    sum_low = _mm256_extractf128_ps(sum1, 0);
    sum_low = _mm_add_ps(sum_low, sum_high);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    re += _mm_cvtss_f32(sum_low);
  }

  // 处理剩余的块
  for (; i <= n - 16; i += 16) {
    _mm_prefetch((const char*)(a + i + 32), _MM_HINT_T0);

    #ifdef __FMA__
      sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), _mm256_setzero_ps());
      sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum1);
      sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), sum2);
    #else
      sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), _mm256_setzero_ps());
      sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), _mm256_setzero_ps());
      sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), _mm256_setzero_ps());
      sum1 = _mm256_add_ps(sum1, sum2);
      sum1 = _mm256_add_ps(sum1, sum3);
    #endif
    
    sum_high = _mm256_extractf128_ps(sum1, 1);
    sum_low = _mm256_extractf128_ps(sum1, 0);
    sum_low = _mm_add_ps(sum_low, sum_high);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    re += _mm_cvtss_f32(sum_low);
  }

  // 处理剩余的 8 个或 4 个元素
  if (i <= n - 8) {  
    sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), _mm256_setzero_ps());
    sum_high = _mm256_extractf128_ps(sum1, 1);
    sum_low = _mm256_extractf128_ps(sum1, 0);
    sum_low = _mm_add_ps(sum_low, sum_high);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    re += _mm_cvtss_f32(sum_low);

    i += 8;
  } else if (i <= n - 4) {  
    sum_low = _mm_fmadd_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i), _mm_setzero_ps());
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    sum_low = _mm_hadd_ps(sum_low, sum_low);
    re += _mm_cvtss_f32(sum_low);

    i += 4;
  }

  // 处理最后剩余的少于 4 个的元素
  for (; i < n; ++i) {
    re += a[i] * b[i];
  }

  return re;
}

int main() {
    float a[1234];
    float b[1234];
    float re = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-0.2f, 0.2f);
    for(int index = 0; index < 1234; index++){
        a[index] = dis(gen);
    }
    for(int index = 0; index < 1234; index++){
        b[index] = dis(gen);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100000; i++){
        re += dot_product(a, b, 1234);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << re << std::endl;
    std::cout << duration.count() << "s" << std::endl;

    return 0;
}