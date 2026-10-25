#include "lab.h"
#include "quad_table.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
extern size_t jeaiii_format(char *, int64_t);
static const char pairs[] =
    "00010203040506070809101112131415161718192021222324252627282930313233343536"
    "37383940414243444546474849505152535455565758596061626364656667686970717273"
    "7475767778798081828384858687888990919293949596979899";
static char quad[40000];
void format_init(void) {
  if (!strcmp(opt.format, "quad") && !opt.static_table)
    for (unsigned i = 0; i < 10000; i++) {
      memcpy(quad + 4 * i, pairs + 2 * (i / 100), 2);
      memcpy(quad + 4 * i + 2, pairs + 2 * (i % 100), 2);
    }
}
size_t format_i64(char *out, int64_t x) {
  if (!strcmp(opt.format, "jeaiii"))
    return jeaiii_format(out, x);
  if (!strcmp(opt.format, "snprintf")) {
    int n = snprintf(out, 32, "%lld", (long long)x);
    if (n < 0)
      fail("format failure");
    return (size_t)n;
  }
  uint64_t u = (uint64_t)x;
  int neg = x < 0;
  if (neg)
    u = 0 - u;
  char buf[32];
  size_t pos = sizeof buf;
  int four = !strcmp(opt.format, "quad");
  if (four)
    while (u >= 10000) {
      unsigned r = (unsigned)(u % 10000);
      u /= 10000;
      pos -= 4;
      memcpy(buf + pos, (opt.static_table ? static_quad : quad) + 4 * r, 4);
    }
  while (u >= 100) {
    unsigned r = (unsigned)(u % 100);
    u /= 100;
    pos -= 2;
    memcpy(buf + pos, pairs + 2 * r, 2);
  }
  if (u >= 10) {
    pos -= 2;
    memcpy(buf + pos, pairs + 2 * u, 2);
  } else
    buf[--pos] = (char)('0' + u);
  if (neg)
    buf[--pos] = '-';
  size_t len = sizeof buf - pos;
  memcpy(out, buf + pos, len);
  return len;
}
static void write_all(int fd, const char *buf, size_t n, uint64_t off) {
  while (n) {
    size_t step = n;
    if (opt.write_chunk && step > opt.write_chunk)
      step = opt.write_chunk;
    if (step > INT_MAX)
      step = INT_MAX;
    ssize_t done = pwrite(fd, buf, step, (off_t)off);
    if (done < 0 && errno == EINTR)
      continue;
    if (done <= 0)
      fail("pwrite: %s", done < 0 ? strerror(errno) : "zero progress");
    buf += done;
    n -= (size_t)done;
    off += (uint64_t)done;
  }
}
#if defined(__x86_64__)
__attribute__((target("avx2"))) static void
vector_decimal(const int64_t *input, int count, char text[8][32], size_t len[8]) {
  int32_t src[8] = {0};
  for (int q = 0; q < count; q++)
    src[q] = (int32_t)input[q];
  __m256i x = _mm256_loadu_si256((const __m256i *)src), sign = _mm256_srai_epi32(x, 31);
  x = _mm256_sub_epi32(_mm256_xor_si256(x, sign), sign);
  unsigned digits[10][8];
  for (int t = 0; t < 10; t++) {
    __m256i magic = _mm256_set1_epi32((int32_t)0xcccccccdU);
    __m256i even = _mm256_srli_epi64(_mm256_mul_epu32(x, magic), 35);
    __m256i odd = _mm256_slli_epi64(
        _mm256_srli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(x, 32), magic), 35), 32);
    __m256i quotient = _mm256_or_si256(even, odd),
            remainder = _mm256_sub_epi32(
                x, _mm256_mullo_epi32(quotient, _mm256_set1_epi32(10)));
    _mm256_storeu_si256((__m256i *)digits[t], remainder);
    x = quotient;
  }
  for (int q = 0; q < count; q++) {
    int top = 9;
    while (top && digits[top][q] == 0)
      top--;
    size_t pos = 0;
    if (src[q] < 0)
      text[q][pos++] = '-';
    for (int t = top; t >= 0; t--)
      text[q][pos++] = (char)('0' + digits[t][q]);
    len[q] = pos;
  }
}
#endif
char *output_begin(int rows, size_t *used) {
  size_t cap = checked_mul(checked_mul(rows, opt.n), 32) + 4;
  char *out = mem_new(cap, 1, 0);
  *used = 0;
  if (rank_id == 0) {
    memcpy(out, "#D\n", 3);
    *used = 3;
  }
  format_init();
  return out;
}
void output_append(char *out, size_t *position, const int64_t *d, int rows, int n,
                   int ld) {
  size_t used = *position;
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < n;) {
      if (!strcmp(opt.format, "batch") || !strcmp(opt.format, "vector")) {
        char tmp[8][32];
        size_t lengths[8], offsets[8];
        int batch = n - j;
        if (batch > 8)
          batch = 8;
        size_t bytes = 0;
        int vector = !strcmp(opt.format, "vector") && has_avx2();
        for (int q = 0; q < batch; q++)
          if (d[(size_t)i * ld + j + q] < INT32_MIN ||
              d[(size_t)i * ld + j + q] > INT32_MAX)
            vector = 0;
#if defined(__x86_64__)
        if (vector)
          vector_decimal(d + (size_t)i * ld + j, batch, tmp, lengths);
#endif
        for (int q = 0; q < batch; q++) {
          offsets[q] = bytes;
          if (!vector)
            lengths[q] = format_i64(tmp[q], d[(size_t)i * ld + j + q]);
          tmp[q][lengths[q]++] = (j + q + 1 == n) ? '\n' : ' ';
          bytes += lengths[q];
        }
        for (int q = 0; q < batch; q++)
          memcpy(out + used + offsets[q], tmp[q], lengths[q]);
        used += bytes;
        j += batch;
      } else {
        used += format_i64(out + used, d[(size_t)i * ld + j]);
        out[used++] = ++j == n ? '\n' : ' ';
      }
    }
  *position = used;
}
void output_finish(char *out, size_t used) {
  uint64_t bytes = used, off = 0, total = 0;
  if (!strcmp(opt.offset, "allgather")) {
    uint64_t *counts = mem_new(world_size, 8, 0);
    MC(MPI_Allgather(&bytes, 1, MPI_UINT64_T, counts, 1, MPI_UINT64_T, MPI_COMM_WORLD));
    for (int r = 0; r < world_size; r++) {
      if (r < rank_id)
        off += counts[r];
      total += counts[r];
    }
    mem_drop(counts);
  } else {
    MC(MPI_Exscan(&bytes, &off, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD));
    if (rank_id == 0)
      off = 0;
    MC(MPI_Allreduce(&bytes, &total, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD));
  }
  if (total > INT64_MAX)
    fail("output exceeds signed file offset range");
  if (!strcmp(opt.writer, "pwrite")) {
    if (rank_id == 0) {
      int fd = open(opt.output, O_CREAT | O_TRUNC | O_WRONLY, 0666);
      if (fd < 0)
        fail("output create: %s", strerror(errno));
      if (ftruncate(fd, (off_t)total) || close(fd))
        fail("output preallocation/close failed");
    }
    MC(MPI_Barrier(MPI_COMM_WORLD));
    int fd = open(opt.output, O_WRONLY);
    if (fd < 0)
      fail("output open failed");
    write_all(fd, out, used, off);
    if (close(fd))
      fail("output close failed");
  } else {
    MPI_File file;
    MC(MPI_File_open(MPI_COMM_WORLD, opt.output, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                     MPI_INFO_NULL, &file));
    MC(MPI_File_set_size(file, (MPI_Offset)total));
    MPI_Status status;
    int count = mpi_count(used);
    if (!strcmp(opt.writer, "mpi-all"))
      MC(MPI_File_write_at_all(file, (MPI_Offset)off, out, count, MPI_BYTE, &status));
    else
      MC(MPI_File_write_at(file, (MPI_Offset)off, out, count, MPI_BYTE, &status));
    int actual;
    MC(MPI_Get_count(&status, MPI_BYTE, &actual));
    if (actual != count)
      fail("short MPI file write");
    MC(MPI_File_close(&file));
  }
  mem_drop(out);
}
void output_write(const int64_t *d, int rows, int n, int ld) {
  size_t used;
  char *out = output_begin(rows, &used);
  output_append(out, &used, d, rows, n, ld);
  output_finish(out, used);
}
