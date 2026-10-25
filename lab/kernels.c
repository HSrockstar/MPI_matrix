#include "lab.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__)
#include <cpuid.h>
#include <immintrin.h>
#endif

int has_avx2(void) {
#if defined(__x86_64__)
  return __builtin_cpu_supports("avx2") != 0;
#else
  return 0;
#endif
}
int has_vnni(void) {
#if defined(__x86_64__)
  unsigned a, b, c, d;
  if (!has_avx2() || __get_cpuid_max(0, NULL) < 7)
    return 0;
  __cpuid_count(7, 0, a, b, c, d);
  if (a < 1)
    return 0;
  __cpuid_count(7, 1, a, b, c, d);
  return (a & (1u << 4)) != 0;
#else
  return 0;
#endif
}
void kernel_validate(int64_t a, int64_t b) {
  if (!strcmp(opt.kernel, "auto"))
    opt.kernel = (a <= 99 && b <= 99) ? (has_vnni()   ? "vnni"
                                         : has_avx2() ? "avx2-sign"
                                                      : "scalar")
                                      : "scalar";
  const char *small[] = {"vnni",     "avx2-sign",         "nibble",
                         "fp32",     "blas-fp32",         "external",
                         "strassen", "strassen-winograd", NULL};
  for (int i = 0; small[i]; i++)
    if (!strcmp(opt.kernel, small[i]) && (a > 99 || b > 99))
      fail("%s requires validated A/B in [-99,99]", opt.kernel);
  if (!strcmp(opt.kernel, "vnni") && !has_vnni())
    fail("AVX-VNNI unavailable");
  if ((!strcmp(opt.kernel, "avx2-sign") || !strcmp(opt.kernel, "nibble")) &&
      !has_avx2())
    fail("AVX2 unavailable");
  if ((!strcmp(opt.kernel, "fp64") || !strcmp(opt.kernel, "blas")) &&
      (__int128)a * b * opt.k >= ((__int128)1 << 53))
    fail("FP64 exactness bound exceeded");
  if (!strcmp(opt.kernel, "winograd") && (__int128)(a + b) * (a + b) * opt.k +
                                                 (__int128)a * a * opt.k +
                                                 (__int128)b * b * opt.k >
                                             INT64_MAX)
    fail("Winograd intermediate bound exceeded");
}
PackedB pack_b(const int32_t *b, int k, int n, int ldb) {
  PackedB p = {0};
  p.k = k;
  p.n = n;
  p.k4 = (k + 3) / 4;
  p.panels = (n + opt.nr - 1) / opt.nr;
  p.bytes = checked_mul(checked_mul(p.panels, p.k4), (size_t)opt.nr * 4);
  p.data = mem_new(p.bytes, 1, 1);
  p.sum = mem_new((size_t)p.panels * opt.nr, sizeof(int64_t), 1);
  for (int j = 0; j < n; j++)
    for (int t = 0; t < k; t++) {
      int32_t x = b[(size_t)t * ldb + j];
      if (x < -99 || x > 99)
        fail("packed B value outside proven domain");
      size_t pos =
          (((size_t)(j / opt.nr) * p.k4 + t / 4) * opt.nr + j % opt.nr) * 4 + t % 4;
      p.data[pos] = (uint8_t)(int8_t)x;
      p.sum[j] += x;
    }
  return p;
}
void packed_free(PackedB *p) {
  mem_drop(p->data);
  mem_drop(p->sum);
  memset(p, 0, sizeof *p);
}
static void progress(MPI_Request *req) {
  if (req && *req != MPI_REQUEST_NULL && opt.progress) {
    static double last;
    if (opt.progress_us) {
      double now = MPI_Wtime();
      if ((now - last) * 1e6 < opt.progress_us)
        return;
      last = now;
    }
    int done;
    MC(MPI_Test(req, &done, MPI_STATUS_IGNORE));
  }
}

#if defined(__x86_64__)
/* Constant tile shapes keep accumulator dimensions visible to the compiler.
   Every ISA-specific instruction is confined to a runtime-gated function. */
#define BLOAD(p)                                                                       \
  (opt.aligned ? _mm256_load_si256((const __m256i *)(p))                               \
               : _mm256_loadu_si256((const __m256i *)(p)))
#define DOT_VNNI(a, b, c) _mm256_dpbusd_epi32((c), (a), (b))
#define DOT_SIGN(a, b, c)                                                              \
  _mm256_add_epi32((c),                                                                \
                   _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(a),          \
                                                          _mm256_sign_epi8((b), (a))), \
                                     _mm256_set1_epi16(1)))
#define DOT_NIBBLE(a, b, c)                                                            \
  _mm256_add_epi32(                                                                    \
      (c),                                                                             \
      _mm256_add_epi32(                                                                \
          _mm256_madd_epi16(                                                           \
              _mm256_maddubs_epi16(_mm256_and_si256((a), _mm256_set1_epi8(15)), (b)),  \
              _mm256_set1_epi16(1)),                                                   \
          _mm256_slli_epi32(                                                           \
              _mm256_madd_epi16(                                                       \
                  _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16((a), 4),     \
                                                        _mm256_set1_epi8(15)),         \
                                       (b)),                                           \
                  _mm256_set1_epi16(1)),                                               \
              4)))
#define MICRO_STEP(R, V, DOT, T)                                                       \
  do {                                                                                 \
    int step_index = (T);                                                              \
    for (int v = 0; v < V; v++) {                                                      \
      __m256i bv = BLOAD(b + ((size_t)step_index * V + v) * 32);                       \
      for (int i = 0; i < R; i++) {                                                    \
        uint32_t x =                                                                   \
            a[inter ? (size_t)step_index * R + i : (size_t)i * astride + step_index];  \
        __m256i av = _mm256_set1_epi32((int32_t)x);                                    \
        acc[i][v] = DOT(av, bv, acc[i][v]);                                            \
      }                                                                                \
    }                                                                                  \
  } while (0)
#define DEFINE_MICRO(NAME, ATTR, R, V, DOT)                                            \
  __attribute__((target(ATTR))) static void NAME##R##_##V(                             \
      const uint32_t *a, const uint8_t *b, int groups, int astride, int inter,         \
      int32_t *out) {                                                                  \
    __m256i acc[R][V];                                                                 \
    for (int i = 0; i < R; i++)                                                        \
      for (int v = 0; v < V; v++)                                                      \
        acc[i][v] = _mm256_setzero_si256();                                            \
    if (opt.pipeline && groups) {                                                      \
      __m256i current[V], next[V];                                                     \
      for (int v = 0; v < V; v++)                                                      \
        current[v] = BLOAD((const __m256i *)(b + (size_t)v * 32));                     \
      for (int t = 0; t < groups; t++) {                                               \
        if (t + 1 < groups)                                                            \
          for (int v = 0; v < V; v++)                                                  \
            next[v] = BLOAD((const __m256i *)(b + ((size_t)(t + 1) * V + v) * 32));    \
        for (int i = 0; i < R; i++) {                                                  \
          uint32_t x = a[inter ? (size_t)t * R + i : (size_t)i * astride + t];         \
          __m256i av = _mm256_set1_epi32((int32_t)x);                                  \
          for (int v = 0; v < V; v++)                                                  \
            acc[i][v] = DOT(av, current[v], acc[i][v]);                                \
        }                                                                              \
        if (t + 1 < groups)                                                            \
          for (int v = 0; v < V; v++)                                                  \
            current[v] = next[v];                                                      \
      }                                                                                \
    } else {                                                                           \
      int base = 0;                                                                    \
      if (opt.unroll == 4)                                                             \
        for (; base + 3 < groups; base += 4) {                                         \
          MICRO_STEP(R, V, DOT, base);                                                 \
          MICRO_STEP(R, V, DOT, base + 1);                                             \
          MICRO_STEP(R, V, DOT, base + 2);                                             \
          MICRO_STEP(R, V, DOT, base + 3);                                             \
        }                                                                              \
      else if (opt.unroll == 2)                                                        \
        for (; base + 1 < groups; base += 2) {                                         \
          MICRO_STEP(R, V, DOT, base);                                                 \
          MICRO_STEP(R, V, DOT, base + 1);                                             \
        }                                                                              \
      for (; base < groups; base++)                                                    \
        MICRO_STEP(R, V, DOT, base);                                                   \
    }                                                                                  \
    for (int i = 0; i < R; i++)                                                        \
      for (int v = 0; v < V; v++)                                                      \
        _mm256_storeu_si256((__m256i *)(out + i * (V * 8) + v * 8), acc[i][v]);        \
  }
#define SHAPES(N, A, D)                                                                \
  DEFINE_MICRO(N, A, 4, 2, D)                                                          \
  DEFINE_MICRO(N, A, 6, 2, D)                                                          \
  DEFINE_MICRO(N, A, 8, 1, D) DEFINE_MICRO(N, A, 4, 3, D)
SHAPES(vnni, "avx2,avxvnni", DOT_VNNI)
SHAPES(sign, "avx2", DOT_SIGN)
SHAPES(nibble, "avx2", DOT_NIBBLE)
#define PICK(N)                                                                        \
  do {                                                                                 \
    if (opt.mr == 4 && opt.nr == 16)                                                   \
      N##4_2(a, b, g, s, inter, out);                                                  \
    else if (opt.mr == 6 && opt.nr == 16)                                              \
      N##6_2(a, b, g, s, inter, out);                                                  \
    else if (opt.mr == 8 && opt.nr == 8)                                               \
      N##8_1(a, b, g, s, inter, out);                                                  \
    else                                                                               \
      N##4_3(a, b, g, s, inter, out);                                                  \
  } while (0)
static void micro(const uint32_t *a, const uint8_t *b, int g, int s, int inter,
                  int32_t *out) {
  if (!strcmp(opt.kernel, "avx2-sign"))
    PICK(sign);
  else if (!strcmp(opt.kernel, "nibble"))
    PICK(nibble);
  else
    PICK(vnni);
}
#endif
static void consume_packed(const uint32_t *ap, int m, int stride, int inter,
                           const PackedB *b, int64_t *d, int ldd, int add,
                           MPI_Request *req) {
  int mr = opt.mr, nr = opt.nr, groups = b->k4;
  int signed_a = !strcmp(opt.kernel, "avx2-sign");
  for (int row0 = 0; row0 < m; row0 += opt.rows) {
    int limit = row0 + opt.rows;
    if (limit > m)
      limit = m;
    for (int p = 0; p < b->panels; p++)
      for (int i = row0; i < limit; i += mr) {
        int32_t out[96];
#if defined(__x86_64__)
        micro(ap + (size_t)i * stride, b->data + (size_t)p * groups * nr * 4, groups,
              stride, inter, out);
#else
        fail("packed SIMD kernel unavailable on this architecture");
#endif
        for (int r = 0; r < mr && i + r < m; r++)
          for (int c = 0; c < nr && p * nr + c < b->n; c++) {
            int j = p * nr + c;
            int64_t value = out[r * nr + c] - (signed_a ? 0 : 128 * b->sum[j]);
            size_t pos = (size_t)(i + r) * ldd + j;
            if (add)
              d[pos] += value;
            else
              d[pos] = value;
          }
      }
    progress(req);
  }
}

void gemm_packed(const int32_t *a, int m, int lda, const PackedB *b, int64_t *d,
                 int ldd, int add, MPI_Request *req) {
  if (!m || !b->n) {
    progress(req);
    return;
  }
  if (!b->k) {
    if (!add)
      for (int i = 0; i < m; i++)
        memset(d + (size_t)i * ldd, 0, (size_t)b->n * sizeof *d);
    progress(req);
    return;
  }

  int mr = opt.mr, groups = b->k4, mp = (m + mr - 1) / mr * mr;
  if ((int64_t)b->k * 227 * 99 > INT32_MAX)
    fail("packed accumulator bound exceeded");
  int stride = groups + opt.pad / 4, inter = !strcmp(opt.layout, "interleaved");
  uint32_t *ap = mem_new((size_t)mp * stride, sizeof(uint32_t), 0);
  int signed_a = !strcmp(opt.kernel, "avx2-sign");
  for (int i = 0; i < mp; i++)
    for (int g = 0; g < groups; g++) {
      uint32_t w = 0;
      for (int t = 0; t < 4; t++) {
        int k = g * 4 + t;
        int32_t x = i < m && k < b->k ? a[(size_t)i * lda + k] : 0;
        if (x < -99 || x > 99)
          fail("packed A outside proven domain");
        uint8_t z = signed_a ? (uint8_t)(int8_t)x : (uint8_t)(x + 128);
        w |= (uint32_t)z << (8 * t);
      }
      size_t pos = inter ? (size_t)(i / mr) * mr * stride + (size_t)g * mr + i % mr
                         : (size_t)i * stride + g;
      ap[pos] = w;
    }
  consume_packed(ap, m, stride, inter, b, d, ldd, add, req);
  mem_drop(ap);
}
void gemm_packed_input(const Input *in, int r0, int m, const PackedB *b, int64_t *d,
                       int ldd) {
  if ((int64_t)b->k * 227 * 99 > INT32_MAX)
    fail("packed accumulator bound exceeded");
  int stride = b->k4 + opt.pad / 4, inter = !strcmp(opt.layout, "interleaved");
  int mp = (m + opt.mr - 1) / opt.mr * opt.mr;
  uint32_t *ap = mem_new((size_t)mp * stride, 4, 0);
  input_pack_a(in, r0, m, 0, b->k, ap, stride, inter, !strcmp(opt.kernel, "avx2-sign"));
  consume_packed(ap, m, stride, inter, b, d, ldd, 0, NULL);
  mem_drop(ap);
}
static void scalar(const int32_t *a, const int32_t *b, int64_t *d, int m, int k, int n,
                   int la, int lb, int ld, int add) {
  if (!add)
    for (int i = 0; i < m; i++)
      memset(d + (size_t)i * ld, 0, (size_t)n * sizeof *d);
  for (int t0 = 0; t0 < k; t0 += opt.kb) {
    int end = t0 + opt.kb;
    if (end > k)
      end = k;
    for (int i = 0; i < m; i++)
      for (int t = t0; t < end; t++) {
        int64_t x = a[(size_t)i * la + t];
        for (int j = 0; j < n; j++)
          d[(size_t)i * ld + j] += x * b[(size_t)t * lb + j];
      }
  }
}
static void rec(const int32_t *a, const int32_t *b, int64_t *d, int m, int k, int n,
                int la, int lb, int ld) {
  if (m <= opt.leaf && k <= opt.leaf && n <= opt.leaf) {
    scalar(a, b, d, m, k, n, la, lb, ld, 1);
    return;
  }
  if (m >= k && m >= n) {
    int q = m / 2;
    rec(a, b, d, q, k, n, la, lb, ld);
    rec(a + (size_t)q * la, b, d + (size_t)q * ld, m - q, k, n, la, lb, ld);
  } else if (n >= k) {
    int q = n / 2;
    rec(a, b, d, m, k, q, la, lb, ld);
    rec(a, b + q, d + q, m, k, n - q, la, lb, ld);
  } else {
    int q = k / 2;
    rec(a, b, d, m, q, n, la, lb, ld);
    rec(a + q, b + (size_t)q * lb, d, m, k - q, n, la, lb, ld);
  }
}
static void floating(const int32_t *a, const int32_t *b, int64_t *d, int m, int k,
                     int n, int la, int lb, int ld, int add, int fp32) {
  if (!add)
    for (int i = 0; i < m; i++)
      memset(d + (size_t)i * ld, 0, (size_t)n * sizeof *d);
  int chunk = fp32 ? 1024 : k;
  if (!chunk)
    return;
  for (int t0 = 0; t0 < k; t0 += chunk) {
    int end = t0 + chunk;
    if (end > k)
      end = k;
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        if (fp32) {
          float v = 0;
          for (int t = t0; t < end; t++)
            v += (float)a[(size_t)i * la + t] * (float)b[(size_t)t * lb + j];
          d[(size_t)i * ld + j] += (int64_t)v;
        } else {
          double v = 0;
          for (int t = t0; t < end; t++)
            v += (double)a[(size_t)i * la + t] * (double)b[(size_t)t * lb + j];
          d[(size_t)i * ld + j] += (int64_t)v;
        }
      }
  }
}
static void winograd(const int32_t *a, const int32_t *b, int64_t *d, int m, int k,
                     int n, int la, int lb, int ld, int add) {
  int64_t *alpha = mem_new(m, 8, 1), *beta = mem_new(n, 8, 1);
  for (int t = 0; t + 1 < k; t += 2) {
    for (int i = 0; i < m; i++)
      alpha[i] += (int64_t)a[(size_t)i * la + t] * a[(size_t)i * la + t + 1];
    for (int j = 0; j < n; j++)
      beta[j] += (int64_t)b[(size_t)t * lb + j] * b[(size_t)(t + 1) * lb + j];
  }
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      int64_t v = -alpha[i] - beta[j];
      for (int t = 0; t + 1 < k; t += 2)
        v += ((int64_t)a[(size_t)i * la + t] + b[(size_t)(t + 1) * lb + j]) *
             ((int64_t)a[(size_t)i * la + t + 1] + b[(size_t)t * lb + j]);
      if (k % 2)
        v += (int64_t)a[(size_t)i * la + k - 1] * b[(size_t)(k - 1) * lb + j];
      if (add)
        d[(size_t)i * ld + j] += v;
      else
        d[(size_t)i * ld + j] = v;
    }
  mem_drop(alpha);
  mem_drop(beta);
}
void gemm(const int32_t *a, const int32_t *b, int64_t *d, int m, int k, int n, int la,
          int lb, int ld, int add, MPI_Request *req) {
  if (!m || !n) {
    progress(req);
    return;
  }
  if (!k) {
    if (!add)
      for (int i = 0; i < m; i++)
        memset(d + (size_t)i * ld, 0, (size_t)n * sizeof *d);
    progress(req);
    return;
  }

  if (!strcmp(opt.kernel, "strassen") || !strcmp(opt.kernel, "strassen-winograd"))
    strassen_gemm(a, b, d, m, k, n, la, lb, ld, add,
                  !strcmp(opt.kernel, "strassen-winograd"));
  else if (!strcmp(opt.kernel, "external"))
    external_gemm(a, b, d, m, k, n, la, lb, ld, add);
  else if (!strcmp(opt.kernel, "blas") || !strcmp(opt.kernel, "blas-fp32"))
    blas_gemm(a, b, d, m, k, n, la, lb, ld, add, !strcmp(opt.kernel, "blas-fp32"));
  else if (!strcmp(opt.kernel, "vnni") || !strcmp(opt.kernel, "avx2-sign") ||
           !strcmp(opt.kernel, "nibble")) {
    PackedB p = pack_b(b, k, n, lb);
    gemm_packed(a, m, la, &p, d, ld, add, req);
    packed_free(&p);
  } else if (!strcmp(opt.kernel, "fp32") || !strcmp(opt.kernel, "fp64"))
    floating(a, b, d, m, k, n, la, lb, ld, add, !strcmp(opt.kernel, "fp32"));
  else if (!strcmp(opt.kernel, "winograd"))
    winograd(a, b, d, m, k, n, la, lb, ld, add);
  else if (!strcmp(opt.kernel, "recursive")) {
    if (!add)
      for (int i = 0; i < m; i++)
        memset(d + (size_t)i * ld, 0, (size_t)n * 8);
    rec(a, b, d, m, k, n, la, lb, ld);
  } else
    scalar(a, b, d, m, k, n, la, lb, ld, add);
  progress(req);
}
