#include "lab.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
/* CBLAS LP64 ABI. Reject detectable ILP64 configuration. Sequential runtime
   is required by the assignment; the test environment uses reference BLAS. */
typedef void (*DGemm)(int, int, int, int, int, int, double, const double *, int,
                      const double *, int, double, double *, int);
typedef void (*SGemm)(int, int, int, int, int, int, float, const float *, int,
                      const float *, int, float, float *, int);
void blas_gemm(const int32_t *a, const int32_t *b, int64_t *d, int m, int k, int n,
               int la, int lb, int ld, int add, int single) {
  if (!m || !n || !k) {
    if (!add)
      for (int i = 0; i < m; i++)
        memset(d + (size_t)i * ld, 0, (size_t)n * 8);
    return;
  }
  const char *path = getenv("MATMUL_BLAS_LIBRARY");
  if (!path)
    path = "libblas.so.3";
  void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    fail("BLAS unavailable: %s", dlerror());
  const char *layer = getenv("MKL_INTERFACE_LAYER");
  if (layer && strstr(layer, "ILP64"))
    fail("ILP64 CBLAS is unsupported");
  void *config_symbol = dlsym(handle, "openblas_get_config");
  if (config_symbol) {
    const char *(*config)(void);
    _Static_assert(sizeof config == sizeof config_symbol, "POSIX dlsym ABI required");
    memcpy(&config, &config_symbol, sizeof config);
    if (strstr(config(), "USE64BITINT"))
      fail("ILP64 OpenBLAS is unsupported");
  }
  void *threads_symbol = dlsym(handle, "openblas_get_num_threads");
  if (threads_symbol) {
    int (*threads)(void);
    memcpy(&threads, &threads_symbol, sizeof threads);
    if (threads() != 1)
      fail("BLAS must use one thread per MPI rank");
  }
  DGemm dg = NULL;
  SGemm sg = NULL;
  void *sym = dlsym(handle, single ? "cblas_sgemm" : "cblas_dgemm");
  if (!sym)
    fail("CBLAS entry point unavailable");
  if (single)
    memcpy(&sg, &sym, sizeof sg);
  else
    memcpy(&dg, &sym, sizeof dg);
  int block = single ? 1024 : k;
  void *aa = mem_new((size_t)m * block, single ? 4 : 8, 0),
       *bb = mem_new((size_t)block * n, single ? 4 : 8, 0),
       *out = mem_new((size_t)m * n, single ? 4 : 8, 0);
  if (!add)
    for (int i = 0; i < m; i++)
      memset(d + (size_t)i * ld, 0, (size_t)n * 8);
  for (int t0 = 0; t0 < k; t0 += block) {
    int kk = k - t0;
    if (kk > block)
      kk = block;
    if (single) {
      float *x = aa, *y = bb, *z = out;
      for (int i = 0; i < m; i++)
        for (int t = 0; t < kk; t++)
          x[(size_t)i * kk + t] = (float)a[(size_t)i * la + t0 + t];
      for (int t = 0; t < kk; t++)
        for (int j = 0; j < n; j++)
          y[(size_t)t * n + j] = (float)b[(size_t)(t0 + t) * lb + j];
      sg(101, 111, 111, m, n, kk, 1, x, kk, y, n, 0, z, n);
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          d[(size_t)i * ld + j] += (int64_t)z[(size_t)i * n + j];
    } else {
      double *x = aa, *y = bb, *z = out;
      for (int i = 0; i < m; i++)
        for (int t = 0; t < kk; t++)
          x[(size_t)i * kk + t] = (double)a[(size_t)i * la + t0 + t];
      for (int t = 0; t < kk; t++)
        for (int j = 0; j < n; j++)
          y[(size_t)t * n + j] = (double)b[(size_t)(t0 + t) * lb + j];
      dg(101, 111, 111, m, n, kk, 1, x, kk, y, n, 0, z, n);
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          d[(size_t)i * ld + j] += (int64_t)z[(size_t)i * n + j];
    }
  }
  mem_drop(aa);
  mem_drop(bb);
  mem_drop(out);
  if (dlclose(handle))
    fail("BLAS dlclose failure");
}
