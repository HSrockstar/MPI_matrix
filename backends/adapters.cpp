#include "backend.h"
#include <algorithm>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>
#if defined(MATRIX_ONEDNN)
#include <oneapi/dnnl/dnnl.hpp>
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SEQ
#error "Build oneDNN with ONEDNN_CPU_RUNTIME=SEQ for this MPI-only project"
#endif
#elif defined(MATRIX_FBGEMM)
#include <fbgemm/Fbgemm.h>
#elif defined(MATRIX_LIBXSMM)
#include <libxsmm.h>
#elif !defined(MATRIX_REFERENCE)
#error "Select exactly one MATRIX_ONEDNN/FBGEMM/LIBXSMM/REFERENCE backend"
#endif

#if defined(MATRIX_ONEDNN) || defined(MATRIX_FBGEMM)
static void dot_u4s8(const uint8_t *a, const int8_t *b, int32_t *d, int m, int k,
                     int n) {
#if defined(MATRIX_ONEDNN)
  using namespace dnnl;
  engine eng(engine::kind::cpu, 0);
  stream queue(eng);
  auto am = memory::desc({m, k}, memory::data_type::u8, memory::format_tag::ab);
  auto bm = memory::desc({k, n}, memory::data_type::s8, memory::format_tag::ab);
  auto dm = memory::desc({m, n}, memory::data_type::s32, memory::format_tag::ab);
  auto pd = matmul::primitive_desc(eng, am, bm, dm);
  auto primitive = matmul(pd);
  primitive.execute(queue,
                    {{DNNL_ARG_SRC, memory(am, eng, const_cast<uint8_t *>(a))},
                     {DNNL_ARG_WEIGHTS, memory(bm, eng, const_cast<int8_t *>(b))},
                     {DNNL_ARG_DST, memory(dm, eng, d)}});
  queue.wait();
#else
  using namespace fbgemm;
  PackBMatrix<int8_t> pb(matrix_op_t::NoTranspose, k, n, b, n, nullptr, 1);
  PackAMatrix<uint8_t> pa(matrix_op_t::NoTranspose, m, k, a, k, nullptr, 1);
  DoNothing<int32_t, int32_t> end;
  memCopy<> copy(end);
  // Explicit one-thread contract. No OpenMP region and no requantization.
  fbgemmPacked(pa, pb, d, d, static_cast<uint32_t>(n), copy, 0, 1);
#endif
}
#endif

extern "C" int matrix_backend_v1(const int32_t *a, const int32_t *b, int64_t *d, int m,
                                 int k, int n, int la, int lb, int ld, int add,
                                 char *error, size_t capacity) {
  try {
    if (m < 0 || k < 0 || n < 0 || la < k || lb < n || ld < n)
      throw std::runtime_error("invalid backend strides");
    for (int i = 0; i < m; i++)
      for (int t = 0; t < k; t++)
        if (a[(size_t)i * la + t] < -99 || a[(size_t)i * la + t] > 99)
          throw std::runtime_error("A outside [-99,99]");
    for (int t = 0; t < k; t++)
      for (int j = 0; j < n; j++)
        if (b[(size_t)t * lb + j] < -99 || b[(size_t)t * lb + j] > 99)
          throw std::runtime_error("B outside [-99,99]");
    if (!add)
      for (int i = 0; i < m; i++)
        std::fill(d + (size_t)i * ld, d + (size_t)i * ld + n, int64_t(0));
    if (!m || !k || !n)
      return 0;
#if defined(MATRIX_REFERENCE)
    for (int i = 0; i < m; i++)
      for (int t = 0; t < k; t++)
        for (int j = 0; j < n; j++)
          d[(size_t)i * ld + j] +=
              int64_t(a[(size_t)i * la + t]) * b[(size_t)t * lb + j];
#elif defined(MATRIX_LIBXSMM)
    libxsmm_init();
    // FP64 blocks: |sum| <= 64*99^2, including any reduction order.
    // Row-major D=A*B is column-major D^T=B^T*A^T.
    for (int i0 = 0; i0 < m; i0 += 64)
      for (int j0 = 0; j0 < n; j0 += 64)
        for (int t0 = 0; t0 < k; t0 += 64) {
          int mm = std::min(64, m - i0), nn = std::min(64, n - j0),
              kk = std::min(64, k - t0);
          std::vector<double> aa((size_t)mm * kk), bb((size_t)kk * nn),
              cc((size_t)mm * nn);
          for (int i = 0; i < mm; i++)
            for (int t = 0; t < kk; t++)
              aa[(size_t)i * kk + t] = a[(size_t)(i + i0) * la + t + t0];
          for (int t = 0; t < kk; t++)
            for (int j = 0; j < nn; j++)
              bb[(size_t)t * nn + j] = b[(size_t)(t + t0) * lb + j + j0];
          auto shape = libxsmm_create_gemm_shape(
              nn, mm, kk, nn, kk, nn, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
              LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64);
          auto kernel = libxsmm_dispatch_gemm(shape, LIBXSMM_GEMM_FLAG_BETA_0,
                                              LIBXSMM_GEMM_PREFETCH_NONE);
          if (!kernel)
            throw std::runtime_error("LIBXSMM dispatch failed");
          libxsmm_gemm_param param = {};
          param.a.primary = bb.data();
          param.b.primary = aa.data();
          param.c.primary = cc.data();
          kernel(&param);
          for (int i = 0; i < mm; i++)
            for (int j = 0; j < nn; j++)
              d[(size_t)(i + i0) * ld + j + j0] += int64_t(cc[(size_t)i * nn + j]);
        }
    // JIT cache lives only within this process. First dispatch remains timed.
#else
    for (int t0 = 0; t0 < k; t0 += 512) {
      int kk = std::min(512, k - t0);
      std::vector<uint8_t> lo((size_t)m * kk), hi((size_t)m * kk);
      std::vector<int8_t> bb((size_t)kk * n);
      std::vector<int64_t> sums(n, 0);
      std::vector<int32_t> low((size_t)m * n), high((size_t)m * n);
      for (int i = 0; i < m; i++)
        for (int t = 0; t < kk; t++) {
          unsigned x = unsigned(a[(size_t)i * la + t + t0] + 128);
          lo[(size_t)i * kk + t] = uint8_t(x & 15);
          hi[(size_t)i * kk + t] = uint8_t(x >> 4);
        }
      for (int t = 0; t < kk; t++)
        for (int j = 0; j < n; j++) {
          int x = b[(size_t)(t + t0) * lb + j];
          bb[(size_t)t * n + j] = int8_t(x);
          sums[j] += x;
        }
      // Each adjacent product pair <=2*15*99; each dot <=512*15*99.
      // Thus AVX2 maddubs saturation and possible FP32 epilogues stay exact.
      dot_u4s8(lo.data(), bb.data(), low.data(), m, kk, n);
      dot_u4s8(hi.data(), bb.data(), high.data(), m, kk, n);
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          d[(size_t)i * ld + j] += int64_t(low[(size_t)i * n + j]) +
                                   16 * int64_t(high[(size_t)i * n + j]) -
                                   128 * sums[j];
    }
#endif
    return 0;
  } catch (const std::exception &e) {
    if (capacity)
      std::snprintf(error, capacity, "%s", e.what());
    return 1;
  } catch (...) {
    if (capacity)
      std::snprintf(error, capacity, "unknown backend exception");
    return 1;
  }
}
