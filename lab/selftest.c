#include "lab.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
static uint32_t state = 1234567;
static int random_value(void) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return (int)(state % 199) - 99;
}
void selftest(void) {
  const int sizes[] = {1, 2, 3, 4, 7, 31, 511, 512, 513, 1023, 2047};
  const char *kernels[] = {"scalar",   "recursive",        "fp32",   "fp64",
                           "winograd", "avx2-sign",        "nibble", "vnni",
                           "strassen", "strassen-winograd"};
  size_t tests = input_selftest();
  Options saved = opt;
  /* Index-coded bytes make lane/per-panel/padding mistakes observable. */
  int32_t coded[9 * 19];
  for (int k = 0; k < 9; k++)
    for (int j = 0; j < 19; j++)
      coded[k * 19 + j] = (k * 37 + j * 11) % 199 - 99;
  PackedB pack = pack_b(coded, 9, 19, 19);
  for (int p = 0; p < pack.panels; p++)
    for (int g = 0; g < pack.k4; g++)
      for (int j = 0; j < opt.nr; j++)
        for (int q = 0; q < 4; q++) {
          int k = g * 4 + q, col = p * opt.nr + j;
          int expected = k < 9 && col < 19 ? coded[k * 19 + col] : 0;
          size_t pos = (((size_t)p * pack.k4 + g) * opt.nr + j) * 4 + q;
          if ((int8_t)pack.data[pos] != expected)
            fail("packed B lane mismatch");
        }
  for (int j = 0; j < pack.panels * opt.nr; j++) {
    int64_t expected = 0;
    if (j < 19)
      for (int k = 0; k < 9; k++)
        expected += coded[k * 19 + j];
    if (pack.sum[j] != expected)
      fail("packed B column sum mismatch");
  }
  packed_free(&pack);
  tests++;

  for (int z = 0; z < 11; z++) {
    int k = sizes[z], m = 7, n = 19;
    int32_t *a = mem_new((size_t)m * k, 4, 0), *b = mem_new((size_t)k * n, 4, 0);
    int64_t *ref = mem_new((size_t)m * n, 8, 0), *out = mem_new((size_t)m * n, 8, 0);
    for (int trial = 0; trial < 3; trial++) {
      for (int i = 0; i < m * k; i++)
        a[i] = trial == 0 ? 99 : trial == 1 ? -99 : random_value();
      for (int i = 0; i < k * n; i++)
        b[i] = trial < 2 ? 99 : random_value();
      opt.kernel = "scalar";
      gemm(a, b, ref, m, k, n, k, n, n, 0, NULL);
      for (int t = 0; t < 10; t++) {
        if (t >= 5 && t <= 7 && !has_avx2())
          continue;
        if (t == 7 && !has_vnni())
          continue;
        opt.kernel = kernels[t];
        for (int l = 0; l < 2; l++) {
          opt.layout = l ? "interleaved" : "rows";
          memset(out, 0xa5, (size_t)m * n * 8);
          gemm(a, b, out, m, k, n, k, n, n, 0, NULL);
          if (memcmp(out, ref, (size_t)m * n * 8))
            fail("selftest mismatch %s K=%d layout=%s", opt.kernel, k, opt.layout);
          tests++;
        }
      }
    }
    mem_drop(a);
    mem_drop(b);
    mem_drop(ref);
    mem_drop(out);
  }
  const int64_t values[] = {INT64_MIN, INT64_MAX, INT32_MIN, INT32_MAX, -10000,  -100,
                            -10,       -1,        0,         1,         9,       10,
                            99,        100,       9999,      10000,     20062647};
  const char *fmts[] = {"pairs", "quad", "jeaiii", "batch", "snprintf"};
  for (int f = 0; f < 5; f++) {
    opt.format = fmts[f];
    format_init();
    for (size_t i = 0; i < sizeof values / sizeof *values; i++) {
      char a[64], b[64];
      memset(a, 'x', sizeof a);
      size_t n = format_i64(a, values[i]);
      snprintf(b, sizeof b, "%lld", (long long)values[i]);
      if (n != strlen(b) || memcmp(a, b, n))
        fail("formatter mismatch %s", opt.format);
      tests++;
    }
  }
  opt = saved;
  mem_finish();
  if (!rank_id)
    printf("SELFTEST PASS %zu kernel/formatter cases per rank\n", tests);
}
