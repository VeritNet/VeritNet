#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <chrono>
#include <immintrin.h>

float dot_product(const float* a, const float* b, int n) {
  __m256 sum = _mm256_setzero_ps();
  int i = 0;

  // AVX2 64
  for (; i <= n - 64; i += 64) {
    _mm_prefetch((const char*)(a + i + 64), _MM_HINT_T0);
    _mm_prefetch((const char*)(b + i + 64), _MM_HINT_T0);

    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum);
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum);
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), sum);
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), sum);
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 32), _mm256_loadu_ps(b + i + 32), sum);
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 40), _mm256_loadu_ps(b + i + 40), sum);
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 48), _mm256_loadu_ps(b + i + 48), sum);
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 56), _mm256_loadu_ps(b + i + 56), sum);
  }

  // AVX2 32
  for (; i <= n - 32; i += 32) {
    _mm_prefetch((const char*)(a + i + 32), _MM_HINT_T0);
    _mm_prefetch((const char*)(b + i + 32), _MM_HINT_T0);

    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum);
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum);
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), sum);
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), sum);
  }

  // AVX2 16
  for (; i <= n - 16; i += 16) {
    _mm_prefetch((const char*)(a + i + 16), _MM_HINT_T0);
    _mm_prefetch((const char*)(b + i + 16), _MM_HINT_T0);

    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum);
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), sum);
  }

  // AVX2 8
  for (; i <= n - 8; i += 8) {
    sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum);
  }

  // SSE2 4
  if (i <= n - 4) {
    __m128 sum4 = _mm_fmadd_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i), _mm_setzero_ps());
    
    // SSE2
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);

    // SSE2 to AVX2
    sum = _mm256_add_ps(sum, _mm256_castps128_ps256(sum4));

    i += 4;
  }

  // AVX2
  __m128 sum_high = _mm256_extractf128_ps(sum, 1);
  __m128 sum_low = _mm256_extractf128_ps(sum, 0);
  sum_low = _mm_add_ps(sum_low, sum_high);
  sum_low = _mm_hadd_ps(sum_low, sum_low);
  sum_low = _mm_hadd_ps(sum_low, sum_low);
  float re = _mm_cvtss_f32(sum_low);

  for (; i < n; ++i) {
    re += a[i] * b[i];
  }

  return re;
}

float dot_product_g(float* lhs, float* rhs, int n) {
    float sum = 0.0f;
    int i = 0;
    for (; i < n-4; i += 4) {
        sum += lhs[i] * rhs[i]; 
        sum += lhs[i + 1] * rhs[i + 1];
        sum += lhs[i + 2] * rhs[i + 2];
        sum += lhs[i + 3] * rhs[i + 3];
    }
    for(;i<n;i++){
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

int main() {
    float a[1234];
    float b[1234];
    float re = 0;
    float re2 = 0;

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
    for(int i = 0; i < 1; i++){
        re += dot_product(a, b, 1234);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto start2 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1; i++){
        re2 += dot_product_g(a, b, 1234);
    }
    auto end2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << re << std::endl;
    std::cout << duration.count() << "s" << std::endl;

    std::chrono::duration<double> duration2 = end2 - start2;
    std::cout << re2 << std::endl;
    std::cout << duration2.count() << "s" << std::endl;

    return 0;
}
