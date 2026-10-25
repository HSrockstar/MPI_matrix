#include "../backends/backend.h"
#include "lab.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
void external_gemm(const int32_t *a, const int32_t *b, int64_t *d, int m, int k, int n,
                   int la, int lb, int ld, int add) {
  const char *path = getenv("MATMUL_BACKEND_LIBRARY");
  if (!path || !*path)
    fail("external kernel requires MATMUL_BACKEND_LIBRARY");
  void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!library)
    fail("backend dlopen: %s", dlerror());
  void *symbol = dlsym(library, "matrix_backend_v1");
  if (!symbol)
    fail("backend missing matrix_backend_v1");
  matrix_backend_fn call;
  _Static_assert(sizeof call == sizeof symbol, "POSIX dlsym ABI required");
  memcpy(&call, &symbol, sizeof call);
  char error[256] = {0};
  if (call(a, b, d, m, k, n, la, lb, ld, add, error, sizeof error))
    fail("backend: %.255s", error);
  if (dlclose(library))
    fail("backend dlclose failed");
}
