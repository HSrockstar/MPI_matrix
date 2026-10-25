#ifndef MATMUL_LAB_H
#define MATMUL_LAB_H
#define _GNU_SOURCE
#include <mpi.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  int m, k, n, kb, mr, nr, rows, pad, unroll, progress, pipeline, leaf, aligned,
      progress_us, memory_stats;
  const char *kernel, *dist, *bmode, *parser, *reader, *prefault, *format, *writer,
      *offset;
  const char *alloc, *layout, *input, *output;
  int beta0, arena, static_table, balance, stream, fused;
  size_t write_chunk;
} Options;
extern Options opt;
extern int rank_id, world_size;
void fail(const char *fmt, ...);
void mpi_check(int code, const char *operation);
#define MC(call) mpi_check((call), #call)
size_t checked_mul(size_t a, size_t b);
int mpi_count(size_t n);
void *mem_new(size_t n, size_t width, int zero);
void mem_drop(void *p);
void mem_finish(void);
int part_size(int n, int p, int r);
int part_start(int n, int p, int r);
typedef struct {
  char *data;
  size_t size;
  uint64_t *off;
  int lines;
  int mapped;
} Input;
Input input_open(const char *path);
void input_close(Input *in);
void input_rows(const Input *in, int matrix, int r0, int nr, int c0, int nc,
                int32_t *out, int ld);
void input_bounds(const Input *in, int64_t bounds[3]);
void input_pack_a(const Input *in, int r0, int rows, int k0, int k, uint32_t *ap,
                  int stride, int inter, int signed_a);
size_t input_selftest(void);
void input_prefault(const Input *in, int matrix, int r0, int nr);
size_t format_i64(char *out, int64_t x);
void format_init(void);
void output_write(const int64_t *d, int rows, int n, int ld);
char *output_begin(int rows, size_t *used);
void output_append(char *out, size_t *used, const int64_t *d, int rows, int n, int ld);
void output_finish(char *out, size_t used);

typedef struct {
  uint8_t *data;
  int64_t *sum;
  int k, n, k4, panels;
  size_t bytes;
} PackedB;
PackedB pack_b(const int32_t *b, int k, int n, int ldb);
void packed_free(PackedB *p);
void gemm_packed(const int32_t *a, int m, int lda, const PackedB *b, int64_t *d,
                 int ldd, int add, MPI_Request *request);
void gemm_packed_input(const Input *in, int r0, int m, const PackedB *b, int64_t *d,
                       int ldd);
void gemm(const int32_t *a, const int32_t *b, int64_t *d, int m, int k, int n, int lda,
          int ldb, int ldd, int add, MPI_Request *request);
int has_avx2(void);
int has_vnni(void);
void blas_gemm(const int32_t *, const int32_t *, int64_t *, int, int, int, int, int,
               int, int, int);
void external_gemm(const int32_t *, const int32_t *, int64_t *, int, int, int, int, int,
                   int, int);
void strassen_gemm(const int32_t *, const int32_t *, int64_t *, int, int, int, int, int,
                   int, int, int);
void kernel_validate(int64_t amax, int64_t bmax);
void run_rows(const Input *in, int64_t **d, int *rows, int *r0, int *ld);
void run_split(const Input *in, int64_t **d, int *rows, int *r0, int *ld);
void run_grid(const Input *in, int64_t **d, int *rows, int *r0, int *ld);
void run_stream(const Input *in);
void selftest(void);
#endif
