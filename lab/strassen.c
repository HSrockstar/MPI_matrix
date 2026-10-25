#include "lab.h"
#include <string.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
/* One level, rectangular padding, linear combinations fused directly into
 * signed-16 pair packing, one product buffer consumed at a time. No int8
 * narrowing of Strassen/Winograd intermediates is permitted. */
static const int su[7][4] = {{1, 0, 0, 1}, {0, 0, 1, 1},  {1, 0, 0, 0}, {0, 0, 0, 1},
                             {1, 1, 0, 0}, {-1, 0, 1, 0}, {0, 1, 0, -1}};
static const int sv[7][4] = {{1, 0, 0, 1}, {1, 0, 0, 0}, {0, 1, 0, -1}, {-1, 0, 1, 0},
                             {0, 0, 0, 1}, {1, 1, 0, 0}, {0, 0, 1, 1}};
static const int sw[4][7] = {{1, 0, 0, 1, -1, 0, 1},
                             {0, 0, 1, 0, 1, 0, 0},
                             {0, 1, 0, 1, 0, 0, 0},
                             {1, -1, 1, 0, 0, 1, 0}};
static const int wu[7][4] = {{1, 0, 0, 0}, {0, 1, 0, 0},  {1, 1, -1, -1}, {0, 0, 0, 1},
                             {0, 0, 1, 1}, {-1, 0, 1, 1}, {1, 0, -1, 0}};
static const int wv[7][4] = {{1, 0, 0, 0},  {0, 0, 1, 0},  {0, 0, 0, 1}, {1, -1, -1, 1},
                             {-1, 1, 0, 0}, {1, -1, 0, 1}, {0, -1, 0, 1}};
static const int ww[4][7] = {{1, 1, 0, 0, 0, 0, 0},
                             {1, 0, 1, 0, 1, 1, 0},
                             {1, 0, 0, -1, 0, 1, 1},
                             {1, 0, 0, 0, 1, 1, 1}};
static int combination(const int32_t *a, int rows, int cols, int ld, int h, int w,
                       int i, int j, const int coefficients[4]) {
  int value = 0;
  for (int block = 0; block < 4; block++)
    if (coefficients[block]) {
      int r = i + (block / 2) * h, c = j + (block % 2) * w;
      if (r < rows && c < cols) {
        int32_t x = a[(size_t)r * ld + c];
        if (x < -99 || x > 99)
          fail("Strassen leaf requires [-99,99] inputs");
        value += coefficients[block] * x;
      }
    }
  return value;
}
#if defined(__x86_64__)
__attribute__((target("avx2"))) static void
pair_micro(const uint32_t *a, const uint32_t *b, int groups, int32_t out[4][8]) {
  __m256i acc[4];
  for (int i = 0; i < 4; i++)
    acc[i] = _mm256_setzero_si256();
  for (int g = 0; g < groups; g++) {
    __m256i bv = _mm256_loadu_si256((const __m256i *)(b + (size_t)g * 8));
    for (int i = 0; i < 4; i++) {
      __m256i av = _mm256_set1_epi32((int32_t)a[(size_t)i * groups + g]);
      acc[i] = _mm256_add_epi32(acc[i], _mm256_madd_epi16(av, bv));
    }
  }
  for (int i = 0; i < 4; i++)
    _mm256_storeu_si256((__m256i *)out[i], acc[i]);
}
#endif
void strassen_gemm(const int32_t *a, const int32_t *b, int64_t *d, int m, int k, int n,
                   int la, int lb, int ld, int add, int winograd) {
  int mh = (m + 1) / 2, kh = (k + 1) / 2, nh = (n + 1) / 2, mp = (mh + 3) / 4 * 4,
      panels = (nh + 7) / 8;
  const int (*u)[4] = winograd ? wu : su, (*v)[4] = winograd ? wv : sv,
            (*w)[7] = winograd ? ww : sw;
  if (!add)
    for (int i = 0; i < m; i++)
      memset(d + (size_t)i * ld, 0, (size_t)n * 8);
  int64_t *product = mem_new((size_t)mh * nh, 8, 0);
  for (int q = 0; q < 7; q++) {
    memset(product, 0, (size_t)mh * nh * 8);
    for (int start = 0; start < kh; start += 8192) {
      int count = kh - start;
      if (count > 8192)
        count = 8192;
      int groups = (count + 1) / 2;
      uint32_t *ap = mem_new((size_t)mp * groups, 4, 1),
               *bp = mem_new((size_t)panels * groups * 8, 4, 1);
      for (int i = 0; i < mh; i++)
        for (int g = 0; g < groups; g++)
          for (int t = 0; t < 2; t++) {
            int index = start + g * 2 + t;
            if (index < start + count) {
              int value = combination(a, m, k, la, mh, kh, i, index, u[q]);
              ap[(size_t)i * groups + g] |= (uint32_t)(uint16_t)(int16_t)value
                                            << (16 * t);
            }
          }
      for (int panel = 0; panel < panels; panel++)
        for (int g = 0; g < groups; g++)
          for (int j = 0; j < 8; j++)
            for (int t = 0; t < 2; t++) {
              int index = start + g * 2 + t, column = panel * 8 + j;
              if (index < start + count && column < nh) {
                int value = combination(b, k, n, lb, kh, nh, index, column, v[q]);
                bp[((size_t)panel * groups + g) * 8 + j] |=
                    (uint32_t)(uint16_t)(int16_t)value << (16 * t);
              }
            }
      // 8192*(4*99)^2 < INT32_MAX, including every partial sum.
      for (int i = 0; i < mh; i += 4)
        for (int panel = 0; panel < panels; panel++) {
          int32_t out[4][8] = {{0}};
#if defined(__x86_64__)
          if (has_avx2())
            pair_micro(ap + (size_t)i * groups, bp + (size_t)panel * groups * 8, groups,
                       out);
          else
#endif
            for (int r = 0; r < 4; r++)
              for (int j = 0; j < 8; j++)
                for (int g = 0; g < groups; g++)
                  for (int t = 0; t < 2; t++) {
                    uint32_t ax = ap[(size_t)(i + r) * groups + g],
                             bx = bp[((size_t)panel * groups + g) * 8 + j];
                    int16_t av = (int16_t)(uint16_t)(ax >> (16 * t)),
                            bv = (int16_t)(uint16_t)(bx >> (16 * t));
                    out[r][j] += (int32_t)av * bv;
                  }
          for (int r = 0; r < 4 && i + r < mh; r++)
            for (int j = 0; j < 8 && panel * 8 + j < nh; j++)
              product[(size_t)(i + r) * nh + panel * 8 + j] += out[r][j];
        }
      mem_drop(ap);
      mem_drop(bp);
    }
    for (int block = 0; block < 4; block++)
      if (w[block][q])
        for (int i = 0; i < mh; i++)
          for (int j = 0; j < nh; j++) {
            int row = i + (block / 2) * mh, col = j + (block % 2) * nh;
            if (row < m && col < n)
              d[(size_t)row * ld + col] +=
                  (int64_t)w[block][q] * product[(size_t)i * nh + j];
          }
  }
  mem_drop(product);
}
