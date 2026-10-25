#include "lab.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

static int dims_rows(int matrix) { return matrix == 1 ? opt.k : opt.m; }
static int dims_cols(int matrix) { return matrix == 0 ? opt.k : opt.n; }
static int first_line(int matrix) {
  return matrix == 0 ? 1 : matrix == 1 ? opt.m + 2 : opt.m + opt.k + 3;
}
static void populate(const Input *in, size_t lo, size_t hi) {
#ifdef MADV_POPULATE_READ
  if (in->mapped && hi > lo) {
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    lo -= lo % page;
    if (madvise(in->data + lo, hi - lo, MADV_POPULATE_READ) && errno != EINVAL &&
        errno != ENOSYS)
      fail("prefault: %s", strerror(errno));
  }
#else
  (void)in;
  (void)lo;
  (void)hi;
#endif
}
void input_prefault(const Input *in, int matrix, int r0, int nr) {
  if (!strcmp(opt.prefault, "local") && nr)
    populate(in, in->off[first_line(matrix) + r0],
             in->off[first_line(matrix) + r0 + nr]);
}
Input input_open(const char *path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0)
    fail("input open: %s", strerror(errno));
  struct stat st, out;
  if (fstat(fd, &st) || st.st_size <= 0)
    fail("invalid input file");
  if (!stat(opt.output, &out) && st.st_dev == out.st_dev && st.st_ino == out.st_ino)
    fail("input and output refer to the same inode");
  Input in = {0};
  in.size = (size_t)st.st_size;
  if (!strcmp(opt.reader, "pread")) {
    in.data = mem_new(in.size, 1, 0);
    size_t done = 0;
    while (done < in.size) {
      ssize_t n = pread(fd, in.data + done, in.size - done, (off_t)done);
      if (n < 0 && errno == EINTR)
        continue;
      if (n <= 0)
        fail("short input read");
      done += (size_t)n;
    }
  } else {
    in.data = mmap(NULL, in.size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (in.data == MAP_FAILED)
      fail("input mmap failed");
    in.mapped = 1;
  }
  if (close(fd))
    fail("input close failed");
  if (!strcmp(opt.prefault, "all"))
    populate(&in, 0, in.size);
  size_t lo = in.size * (size_t)rank_id / (size_t)world_size,
         hi = in.size * (size_t)(rank_id + 1) / (size_t)world_size;
  if (!strcmp(opt.prefault, "local"))
    populate(&in, lo, hi);
  in.lines = 2 * opt.m + opt.k + 3;
  uint64_t *local = mem_new((size_t)in.lines + 1, sizeof *local, 0);
  int count = 0;
  for (size_t i = lo; i < hi; i++)
    if (in.data[i] == '\n') {
      if (count >= in.lines)
        fail("too many lines");
      local[count++] = (uint64_t)i + 1;
    }
  if (rank_id == world_size - 1 && in.data[in.size - 1] != '\n') {
    if (count >= in.lines)
      fail("too many lines");
    local[count++] = (uint64_t)in.size;
  }
  int *counts = mem_new(world_size, sizeof(int), 0),
      *disps = mem_new(world_size, sizeof(int), 0);
  MC(MPI_Allgather(&count, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD));
  int total = 0;
  for (int r = 0; r < world_size; r++) {
    disps[r] = total;
    if (counts[r] > in.lines - total)
      fail("too many lines");
    total += counts[r];
  }
  if (total != in.lines)
    fail("expected %d lines, found %d", in.lines, total);
  in.off = mem_new((size_t)total + 1, sizeof(uint64_t), 0);
  in.off[0] = 0;
  MC(MPI_Allgatherv(local, count, MPI_UINT64_T, in.off + 1, counts, disps, MPI_UINT64_T,
                    MPI_COMM_WORLD));
  for (int x = 0; x < 3; x++) {
    int line = first_line(x) - 1;
    size_t a = in.off[line], b = in.off[line + 1];
    while (b > a && (in.data[b - 1] == '\n' || in.data[b - 1] == '\r'))
      b--;
    if (b - a != 2 || in.data[a] != '#' || in.data[a + 1] != 'A' + x)
      fail("bad matrix marker");
  }
  mem_drop(local);
  mem_drop(counts);
  mem_drop(disps);
  return in;
}

/* SIMD classification of complete short tokens. Narrowing is never performed
   until validation, and no load crosses the current row boundary. */
#if defined(__x86_64__)
static unsigned char compress_table[256][16];
static int compress_ready;
static void init_compress(void) {
  if (compress_ready)
    return;
  for (unsigned mask = 0; mask < 256; mask++) {
    int pos = 0;
    for (int i = 0; i < 8; i++)
      if (mask & (1u << i)) {
        compress_table[mask][pos++] = (unsigned char)(2 * i);
        compress_table[mask][pos++] = (unsigned char)(2 * i + 1);
      }
    while (pos < 16)
      compress_table[mask][pos++] = 128;
  }
  compress_ready = 1;
}
__attribute__((target("avx2"))) static int
short_batch(const char *p, const char *end, int32_t *values, int *used, int width) {
  if (end - p < width + 1)
    return 0;
  init_compress();
  unsigned mask;
  if (width == 32) {
    __m256i wide = _mm256_loadu_si256((const __m256i *)p);
    __m256i digits = _mm256_and_si256(_mm256_cmpgt_epi8(wide, _mm256_set1_epi8('/')),
                                      _mm256_cmpgt_epi8(_mm256_set1_epi8(':'), wide));
    mask = (unsigned)_mm256_movemask_epi8(digits);
  } else {
    __m128i x = _mm_loadu_si128((const __m128i *)p);
    __m128i digits = _mm_and_si128(_mm_cmpgt_epi8(x, _mm_set1_epi8('/')),
                                   _mm_cmpgt_epi8(_mm_set1_epi8(':'), x));
    mask = (unsigned)_mm_movemask_epi8(digits);
  }
  if (mask == 0)
    return 0;
  int pos = 0, n = 0;
  unsigned starts = 0;
  while (pos < width) {
    while (pos < width && (p[pos] == ' ' || p[pos] == '\t'))
      pos++;
    int start = pos;
    if (pos < width && p[pos] == '-')
      pos++;
    int digit_start = pos, d = 0;
    while (pos < width && (mask & (1u << pos))) {
      pos++;
      d++;
      if (d > 2)
        return 0;
    }
    if (!d || pos == width) {
      pos = start;
      break;
    }
    if (p[pos] != ' ' && p[pos] != '\t' && p[pos] != '\r' && p[pos] != '\n')
      return 0;
    starts |= 1u << digit_start;
    n++;
  }
  if (!n)
    return 0;
  int produced = 0;
  for (int offset = 0; offset < width; offset += 16) {
    __m128i x = _mm_loadu_si128((const __m128i *)(p + offset));
    __m128i next = _mm_loadu_si128((const __m128i *)(p + offset + 1));
    __m128i next_digit = _mm_and_si128(_mm_cmpgt_epi8(next, _mm_set1_epi8('/')),
                                       _mm_cmpgt_epi8(_mm_set1_epi8(':'), next));
    __m256i v0 = _mm256_cvtepu8_epi16(_mm_sub_epi8(x, _mm_set1_epi8('0'))),
            v1 = _mm256_cvtepu8_epi16(_mm_sub_epi8(next, _mm_set1_epi8('0')));
    __m256i combined =
        _mm256_add_epi16(_mm256_mullo_epi16(v0, _mm256_set1_epi16(10)), v1);
    __m256i vals = _mm256_blendv_epi8(v0, combined, _mm256_cvtepi8_epi16(next_digit));
    __m128i signbytes = _mm_slli_si128(_mm_cmpeq_epi8(x, _mm_set1_epi8('-')), 1);
    if (offset && p[offset - 1] == '-')
      signbytes = _mm_or_si128(signbytes, _mm_cvtsi32_si128(255));
    __m256i sign = _mm256_cvtepi8_epi16(signbytes);
    vals = _mm256_sub_epi16(_mm256_xor_si256(vals, sign), sign);
    unsigned bits = (starts >> offset) & 65535u;
    unsigned lo = bits & 255, hi = bits >> 8;
    int count = __builtin_popcount(lo);
    int16_t compact[16] = {0};
    _mm_storeu_si128(
        (__m128i *)compact,
        _mm_shuffle_epi8(_mm256_castsi256_si128(vals),
                         _mm_loadu_si128((const __m128i *)compress_table[lo])));
    _mm_storeu_si128(
        (__m128i *)(compact + count),
        _mm_shuffle_epi8(_mm256_extracti128_si256(vals, 1),
                         _mm_loadu_si128((const __m128i *)compress_table[hi])));
    _mm256_storeu_si256(
        (__m256i *)(values + produced),
        _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)compact)));
    _mm256_storeu_si256(
        (__m256i *)(values + produced + 8),
        _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)(compact + 8))));
    produced += __builtin_popcount(bits);
  }
  *used = pos;
  return n;
}
#endif
static int32_t number(const char **p, const char *end) {
  int neg = 0;
  if (*p < end && **p == '-') {
    neg = 1;
    (*p)++;
  }
  if (*p == end || **p < '0' || **p > '9')
    fail("invalid integer token");
  if (!strcmp(opt.parser, "swar") && end - *p >= 3 && (*p)[1] >= '0' &&
      (*p)[1] <= '9' &&
      ((*p)[2] == ' ' || (*p)[2] == '\t' || (*p)[2] == '\r' || (*p)[2] == '\n')) {
    uint16_t pair;
    memcpy(&pair, *p, 2);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    int32_t value = (int32_t)((pair & 15) * 10 + ((pair >> 8) & 15));
    *p += 2;
    return neg ? -value : value;
#endif
  }
  uint64_t v = 0, limit = neg ? 2147483648ULL : 2147483647ULL;
  while (*p < end && **p >= '0' && **p <= '9') {
    unsigned d = (unsigned)(**p - '0');
    if (v > (limit - d) / 10)
      fail("input exceeds int32 range");
    v = v * 10 + d;
    (*p)++;
  }
  if (*p < end && **p != ' ' && **p != '\t' && **p != '\r' && **p != '\n')
    fail("invalid token suffix");
  return neg ? (int32_t)(-(int64_t)v) : (int32_t)v;
}
static void row(const Input *in, int matrix, int r, int c0, int nc, int32_t *out,
                int64_t *bound, uint32_t *packed, int pack_step, int signed_a) {
  int line = first_line(matrix) + r, cols = dims_cols(matrix), col = 0;
  const char *p = in->data + in->off[line], *end = in->data + in->off[line + 1];
  while (p < end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n'))
      p++;
    if (p == end)
      break;
    int32_t v[32];
    int count = 0, used = 0;
#if defined(__x86_64__)
    if ((!strcmp(opt.parser, "simd") || !strcmp(opt.parser, "wide")) && has_avx2())
      count = short_batch(p, end, v, &used, !strcmp(opt.parser, "wide") ? 32 : 16);
#endif
    if (count)
      p += used;
    else {
      v[0] = number(&p, end);
      count = 1;
    }
    for (int t = 0; t < count; t++) {
      if (col >= cols)
        fail("too many values in matrix %d row %d", matrix, r);
      if (out && col >= c0 && col < c0 + nc)
        out[col - c0] = v[t];
      if (packed && col >= c0 && col < c0 + nc) {
        if (v[t] < -99 || v[t] > 99)
          fail("packed input outside [-99,99]");
        int q = col - c0, shift = 8 * (q % 4);
        uint8_t byte = signed_a ? (uint8_t)(int8_t)v[t] : (uint8_t)(v[t] + 128);
        uint32_t *word = packed + (size_t)(q / 4) * pack_step;
        *word = (*word & ~((uint32_t)255 << shift)) | ((uint32_t)byte << shift);
      }
      if (bound) {
        int64_t a = v[t];
        if (a < 0)
          a = -a;
        if (a > *bound)
          *bound = a;
      }
      col++;
    }
  }
  if (col != cols)
    fail("matrix %d row %d: expected %d values, got %d", matrix, r, cols, col);
}
void input_rows(const Input *in, int matrix, int r0, int nr, int c0, int nc,
                int32_t *out, int ld) {
  input_prefault(in, matrix, r0, nr);
  for (int i = 0; i < nr; i++)
    row(in, matrix, r0 + i, c0, nc, out + (size_t)i * ld, NULL, NULL, 0, 0);
}
void input_bounds(const Input *in, int64_t bounds[3]) {
  int64_t local[3] = {0};
  for (int x = 0; x < 3; x++) {
    int nr = dims_rows(x);
    int start = part_start(nr, world_size, rank_id),
        count = part_size(nr, world_size, rank_id);
    for (int i = 0; i < count; i++)
      row(in, x, start + i, 0, 0, NULL, &local[x], NULL, 0, 0);
  }
  MC(MPI_Allreduce(local, bounds, 3, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD));
  __int128 bound = (__int128)bounds[0] * bounds[1] * opt.k + bounds[2];
  if (bound > INT64_MAX)
    fail("cannot guarantee exact int64 accumulation");
}
void input_close(Input *in) {
  if (in->mapped)
    munmap(in->data, in->size);
  else
    mem_drop(in->data);
  mem_drop(in->off);
}

void input_pack_a(const Input *in, int r0, int rows, int k0, int k, uint32_t *ap,
                  int stride, int inter, int signed_a) {
  int mp = (rows + opt.mr - 1) / opt.mr * opt.mr;
  for (size_t q = 0; q < (size_t)mp * stride; q++)
    ap[q] = signed_a ? 0u : 0x80808080u;
  input_prefault(in, 0, r0, rows);
  for (int i = 0; i < rows; i++) {
    size_t offset = inter ? (size_t)(i / opt.mr) * opt.mr * stride + i % opt.mr
                          : (size_t)i * stride;
    row(in, 0, r0 + i, k0, k, NULL, NULL, ap + offset, inter ? opt.mr : 1, signed_a);
  }
}
#if defined(__x86_64__)
__attribute__((target("avx2"))) static size_t check_compress(void) {
  init_compress();
  const int16_t src[8] = {-99, 98, -17, 6, 0, -8, 25, 43};
  for (unsigned mask = 0; mask < 256; mask++) {
    int16_t dst[8];
    _mm_storeu_si128(
        (__m128i *)dst,
        _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)src),
                         _mm_loadu_si128((const __m128i *)compress_table[mask])));
    int count = 0;
    for (int i = 0; i < 8; i++)
      if (mask & (1u << i)) {
        if (dst[count++] != src[i])
          fail("shuffle mask %u order mismatch", mask);
      }
    while (count < 8)
      if (dst[count++] != 0)
        fail("shuffle mask padding mismatch");
  }
  return 256;
}
#endif
size_t input_selftest(void) {
  Options saved = opt;
  size_t tests = 0;
#if defined(__x86_64__)
  if (has_avx2())
    tests += check_compress();
#endif
  char text[4096];
  int32_t actual[96], expected[96];
  uint64_t off[3] = {0, 0, 0};
  Input in = {.data = text, .off = off};
  const char *parsers[] = {"scalar", "simd", "swar", "wide"};
  for (int length = 1; length <= 96; length++) {
    size_t used = 0;
    for (int i = 0; i < length; i++) {
      int value = ((i * 79 + length * 17) % 199) - 99;
      if (i % 19 == 0)
        value = INT32_MIN;
      if (i % 23 == 0)
        value = INT32_MAX;
      expected[i] = value;
      used += (size_t)snprintf(text + used, sizeof text - used, "%d%s", value,
                               i + 1 == length ? "\r\n"
                               : i % 2         ? " "
                                               : "\t");
    }
    off[2] = used;
    opt.k = length;
    for (int mode = 0; mode < 4; mode++) {
      opt.parser = parsers[mode];
      row(&in, 0, 0, 0, length, actual, NULL, NULL, 0, 0);
      if (memcmp(actual, expected, (size_t)length * sizeof *actual))
        fail("parser selftest %s length %d", opt.parser, length);
      tests++;
    }
  }
  opt = saved;
  return tests;
}
