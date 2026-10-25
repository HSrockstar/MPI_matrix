#include "lab.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int start, k;
  int *count, *disp;
  void *data;
  MPI_Request req;
  int packed;
} Stripe;
static int is_packed_kernel(void) {
  return !strcmp(opt.kernel, "vnni") || !strcmp(opt.kernel, "avx2-sign") ||
         !strcmp(opt.kernel, "nibble");
}
static size_t packed_bytes(int k, int n) {
  return checked_mul(checked_mul((k + 3) / 4, (n + opt.nr - 1) / opt.nr),
                     (size_t)opt.nr * 4);
}
static void stripe_start(Stripe *s, const Input *in, int start, int k) {
  memset(s, 0, sizeof *s);
  s->start = start;
  s->k = k;
  s->req = MPI_REQUEST_NULL;
  s->packed = !strcmp(opt.bmode, "packed");
  s->count = mem_new(world_size, sizeof(int), 0);
  s->disp = mem_new(world_size, sizeof(int), 0);
  if (s->packed) {
    if (!is_packed_kernel())
      fail("packed communication requires an int8 SIMD kernel");
    int local = part_size(k, world_size, rank_id),
        first = part_start(k, world_size, rank_id),
        padded = (opt.n + opt.nr - 1) / opt.nr * opt.nr;
    size_t total = 0;
    for (int r = 0; r < world_size; r++) {
      int kr = part_size(k, world_size, r);
      size_t bytes = packed_bytes(kr, opt.n) + (size_t)padded * 8;
      s->disp[r] = mpi_count(total);
      s->count[r] = mpi_count(bytes);
      total += bytes;
    }
    s->data = mem_new(total, 1, 1);
    int32_t *b = mem_new((size_t)local * opt.n, 4, 0);
    input_rows(in, 1, start + first, local, 0, opt.n, b, opt.n);
    PackedB p = pack_b(b, local, opt.n, opt.n);
    char *dst = (char *)s->data + s->disp[rank_id];
    memcpy(dst, p.data, p.bytes);
    memcpy(dst + p.bytes, p.sum, (size_t)padded * 8);
    packed_free(&p);
    mem_drop(b);
    MC(MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_BYTE, s->data, s->count, s->disp, MPI_BYTE,
                       MPI_COMM_WORLD, &s->req));
  } else {
    int equal = !strcmp(opt.bmode, "equal"),
        maxrows = (k + world_size - 1) / world_size;
    int padded = equal ? maxrows * world_size : k;
    s->data = mem_new((size_t)padded * opt.n, 4, 1);
    if (!strcmp(opt.bmode, "replicate")) {
      input_rows(in, 1, start, k, 0, opt.n, s->data, opt.n);
      return;
    }
    if (!strcmp(opt.bmode, "root")) {
      if (rank_id == 0)
        input_rows(in, 1, start, k, 0, opt.n, s->data, opt.n);
      MC(MPI_Ibcast(s->data, mpi_count((size_t)k * opt.n), MPI_INT32_T, 0,
                    MPI_COMM_WORLD, &s->req));
      return;
    }
    for (int r = 0; r < world_size; r++) {
      int nr = equal ? maxrows : part_size(k, world_size, r),
          r0 = equal ? r * maxrows : part_start(k, world_size, r);
      s->count[r] = mpi_count((size_t)nr * opt.n);
      s->disp[r] = mpi_count((size_t)r0 * opt.n);
    }
    int first = equal ? rank_id * maxrows : part_start(k, world_size, rank_id),
        nr = equal ? maxrows : part_size(k, world_size, rank_id);
    if (first + nr > k)
      nr = k - first;
    if (nr < 0)
      nr = 0;
    input_rows(in, 1, start + first, nr, 0, opt.n,
               (int32_t *)s->data + s->disp[rank_id], opt.n);
    if (equal)
      MC(MPI_Iallgather(MPI_IN_PLACE, 0, MPI_INT32_T, s->data, s->count[rank_id],
                        MPI_INT32_T, MPI_COMM_WORLD, &s->req));
    else
      MC(MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_INT32_T, s->data, s->count, s->disp,
                         MPI_INT32_T, MPI_COMM_WORLD, &s->req));
  }
}
static void stripe_drop(Stripe *s) {
  if (s->req != MPI_REQUEST_NULL)
    fail("freeing an in-flight stripe");
  mem_drop(s->data);
  mem_drop(s->count);
  mem_drop(s->disp);
}
static void add_c(const Input *in, int r0, int rows, int64_t *d, int ld) {
  int32_t *c = mem_new((size_t)rows * opt.n, 4, 0);
  input_rows(in, 2, r0, rows, 0, opt.n, c, opt.n);
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < opt.n; j++)
      d[(size_t)i * ld + j] += c[(size_t)i * opt.n + j];
  mem_drop(c);
}
static void row_partition(const Input *in, int *rows, int *r0) {
  if (!opt.balance) {
    *rows = part_size(opt.m, world_size, rank_id);
    *r0 = part_start(opt.m, world_size, rank_id);
    return;
  }
  long double total = 0;
  for (int i = 0; i < opt.m; i++) {
    uint64_t bytes = in->off[i + 2] - in->off[i + 1] + in->off[opt.m + opt.k + 4 + i] -
                     in->off[opt.m + opt.k + 3 + i];
    total += (long double)opt.k * opt.n + (long double)opt.balance * bytes;
  }
  long double low = total * rank_id / world_size,
              high = total * (rank_id + 1) / world_size, prefix = 0;
  int first = 0, last = opt.m;
  for (int i = 0; i < opt.m; i++) {
    if (prefix < low)
      first = i + 1;
    if (prefix >= high) {
      last = i;
      break;
    }
    uint64_t bytes = in->off[i + 2] - in->off[i + 1] + in->off[opt.m + opt.k + 4 + i] -
                     in->off[opt.m + opt.k + 3 + i];
    prefix += (long double)opt.k * opt.n + (long double)opt.balance * bytes;
  }
  *r0 = first;
  *rows = last - first;
}
static void alloc_result(const Input *in, int64_t **d, int *rows, int *r0, int *ld) {
  row_partition(in, rows, r0);
  *ld = opt.n + opt.pad / 8;
  *d = mem_new((size_t)*rows * *ld, 8, !opt.beta0);
}
void run_stream(const Input *in) {
  int rows, r0;
  row_partition(in, &rows, &r0);
  int root_packed = !strcmp(opt.bmode, "root-packed");
  if ((root_packed || opt.fused) && !is_packed_kernel())
    fail("root-packed/fused requires an int8 SIMD kernel");
  int32_t *b = NULL;
  if (!root_packed || !rank_id)
    b = mem_new((size_t)opt.k * opt.n, 4, 0);
  if (!strcmp(opt.bmode, "replicate") || !rank_id)
    input_rows(in, 1, 0, opt.k, 0, opt.n, b, opt.n);
  if (!strcmp(opt.bmode, "root"))
    MC(MPI_Bcast(b, mpi_count((size_t)opt.k * opt.n), MPI_INT32_T, 0, MPI_COMM_WORLD));
  PackedB packed = {0};
  if (is_packed_kernel()) {
    if (!root_packed || !rank_id)
      packed = pack_b(b, opt.k, opt.n, opt.n);
    else {
      packed = (PackedB){.k = opt.k,
                         .n = opt.n,
                         .k4 = (opt.k + 3) / 4,
                         .panels = (opt.n + opt.nr - 1) / opt.nr};
      packed.bytes = packed_bytes(opt.k, opt.n);
      packed.data = mem_new(packed.bytes, 1, 0);
      packed.sum = mem_new((size_t)packed.panels * opt.nr, 8, 0);
    }
    if (root_packed) {
      MC(MPI_Bcast(packed.data, mpi_count(packed.bytes), MPI_BYTE, 0, MPI_COMM_WORLD));
      MC(MPI_Bcast(packed.sum, packed.panels * opt.nr, MPI_INT64_T, 0, MPI_COMM_WORLD));
    }
    mem_drop(b);
    b = NULL;
  }
  int32_t *a = opt.fused ? NULL : mem_new((size_t)opt.rows * opt.k, 4, 0);
  int64_t *d = mem_new((size_t)opt.rows * opt.n, 8, 0);
  size_t used;
  char *out = output_begin(rows, &used);
  for (int i = 0; i < rows; i += opt.rows) {
    int count = rows - i;
    if (count > opt.rows)
      count = opt.rows;
    if (!opt.fused)
      input_rows(in, 0, r0 + i, count, 0, opt.k, a, opt.k);
    if (opt.fused)
      gemm_packed_input(in, r0 + i, count, &packed, d, opt.n);
    else if (is_packed_kernel())
      gemm_packed(a, count, opt.k, &packed, d, opt.n, 0, NULL);
    else
      gemm(a, b, d, count, opt.k, opt.n, opt.k, opt.n, opt.n, 0, NULL);
    add_c(in, r0 + i, count, d, opt.n);
    output_append(out, &used, d, count, opt.n, opt.n);
  }
  output_finish(out, used);
  if (is_packed_kernel())
    packed_free(&packed);
  mem_drop(a);
  mem_drop(b);
  mem_drop(d);
}
void run_rows(const Input *in, int64_t **d, int *rows, int *r0, int *ld) {
  alloc_result(in, d, rows, r0, ld);
  int la = opt.k + opt.pad / 4;
  int32_t *a = mem_new((size_t)*rows * la, 4, 1);
  input_rows(in, 0, *r0, *rows, 0, opt.k, a, la);
  if (!strcmp(opt.dist, "ring")) {
    int maxk = (opt.k + world_size - 1) / world_size;
    int32_t *buf[2] = {mem_new((size_t)maxk * opt.n, 4, 1),
                       mem_new((size_t)maxk * opt.n, 4, 1)};
    int owner = rank_id;
    input_rows(in, 1, part_start(opt.k, world_size, owner),
               part_size(opt.k, world_size, owner), 0, opt.n, buf[0], opt.n);
    for (int step = 0; step < world_size; step++) {
      int slot = step % 2, kr = part_size(opt.k, world_size, owner),
          k0 = part_start(opt.k, world_size, owner);
      MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
      if (step + 1 < world_size) {
        int next = (owner + world_size - 1) % world_size;
        MC(MPI_Irecv(buf[1 - slot],
                     mpi_count((size_t)part_size(opt.k, world_size, next) * opt.n),
                     MPI_INT32_T, (rank_id + world_size - 1) % world_size, 77,
                     MPI_COMM_WORLD, &reqs[0]));
        MC(MPI_Isend(buf[slot], mpi_count((size_t)kr * opt.n), MPI_INT32_T,
                     (rank_id + 1) % world_size, 77, MPI_COMM_WORLD, &reqs[1]));
      }
      gemm(*rows ? a + k0 : a, buf[slot], *d, *rows, kr, opt.n, la, opt.n, *ld,
           !opt.beta0 || step != 0, &reqs[0]);
      MC(MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE));
      owner = (owner + world_size - 1) % world_size;
    }
    mem_drop(buf[0]);
    mem_drop(buf[1]);
  } else {
    Stripe s[2];
    int ns = (opt.k + opt.kb - 1) / opt.kb;
    stripe_start(&s[0], in, 0, opt.k < opt.kb ? opt.k : opt.kb);
    for (int step = 0; step < ns; step++) {
      int slot = step % 2;
      Stripe *cur = &s[slot];
      if (step + 1 < ns) {
        int st = (step + 1) * opt.kb, kk = opt.k - st;
        if (kk > opt.kb)
          kk = opt.kb;
        stripe_start(&s[1 - slot], in, st, kk);
      }
      MC(MPI_Wait(&cur->req, MPI_STATUS_IGNORE));
      MPI_Request *next = step + 1 < ns ? &s[1 - slot].req : NULL;
      if (cur->packed) {
        for (int r = 0; r < world_size; r++) {
          int kr = part_size(cur->k, world_size, r),
              k0 = cur->start + part_start(cur->k, world_size, r);
          int panels = (opt.n + opt.nr - 1) / opt.nr;
          PackedB p = {0};
          p.k = kr;
          p.k4 = (kr + 3) / 4;
          p.n = opt.n;
          p.panels = panels;
          p.bytes = packed_bytes(kr, opt.n);
          p.data = (uint8_t *)cur->data + cur->disp[r];
          p.sum = mem_new((size_t)panels * opt.nr, 8, 0);
          memcpy(p.sum, p.data + p.bytes, (size_t)panels * opt.nr * 8);
          gemm_packed(*rows ? a + k0 : a, *rows, la, &p, *d, *ld,
                      !opt.beta0 || step != 0 || r != 0, next);
          mem_drop(p.sum);
        }
      } else
        gemm(*rows ? a + cur->start : a, cur->data, *d, *rows, cur->k, opt.n, la, opt.n,
             *ld, !opt.beta0 || step != 0, next);
      stripe_drop(cur);
    }
  }
  add_c(in, *r0, *rows, *d, *ld);
  mem_drop(a);
}
void run_split(const Input *in, int64_t **d, int *rows, int *r0, int *ld) {
  int kk = part_size(opt.k, world_size, rank_id),
      k0 = part_start(opt.k, world_size, rank_id);
  int32_t *a = mem_new((size_t)opt.m * kk, 4, 0),
          *b = mem_new((size_t)kk * opt.n, 4, 0);
  input_rows(in, 0, 0, opt.m, k0, kk, a, kk);
  input_rows(in, 1, k0, kk, 0, opt.n, b, opt.n);
  int64_t *tmp = mem_new((size_t)opt.m * opt.n, 8, 0);
  gemm(a, b, tmp, opt.m, kk, opt.n, kk, opt.n, opt.n, 0, NULL);
  *rows = part_size(opt.m, world_size, rank_id);
  *r0 = part_start(opt.m, world_size, rank_id);
  *ld = opt.n;
  *d = mem_new((size_t)*rows * opt.n, 8, 0);
  int *cnt = mem_new(world_size, 4, 0);
  for (int r = 0; r < world_size; r++)
    cnt[r] = mpi_count((size_t)part_size(opt.m, world_size, r) * opt.n);
  MC(MPI_Reduce_scatter(tmp, *d, cnt, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD));
  add_c(in, *r0, *rows, *d, *ld);
  mem_drop(a);
  mem_drop(b);
  mem_drop(tmp);
  mem_drop(cnt);
}

/* SUMMA on any rectangular process grid. A and B panels are read only by the
   selected owners, then broadcast in row/column communicators. Final row
   redistribution uses Alltoallv, not a replicated full-result reduction. */
void run_grid(const Input *in, int64_t **d, int *rows, int *r0, int *ld) {
  int dims[2] = {0, 0};
  if (!strcmp(opt.dist, "columns")) {
    dims[0] = 1;
    dims[1] = world_size;
  } else
    MC(MPI_Dims_create(world_size, 2, dims));
  int pr = dims[0], pc = dims[1], rr = rank_id / pc, cc = rank_id % pc;
  MPI_Comm rowcomm, colcomm;
  MC(MPI_Comm_split(MPI_COMM_WORLD, rr, cc, &rowcomm));
  MC(MPI_Comm_split(MPI_COMM_WORLD, cc, rr, &colcomm));
  int mm = part_size(opt.m, pr, rr), mi = part_start(opt.m, pr, rr),
      nn = part_size(opt.n, pc, cc), nj = part_start(opt.n, pc, cc);
  int64_t *tile = mem_new((size_t)mm * nn, 8, 0);
  int32_t *a = mem_new((size_t)mm * opt.kb, 4, 0),
          *b = mem_new((size_t)opt.kb * nn, 4, 0);
  for (int st = 0, step = 0; st < opt.k; st += opt.kb, step++) {
    int kk = opt.k - st;
    if (kk > opt.kb)
      kk = opt.kb;
    int ar = step % pc, br = step % pr;
    if (cc == ar)
      input_rows(in, 0, mi, mm, st, kk, a, kk);
    if (rr == br)
      input_rows(in, 1, st, kk, nj, nn, b, nn);
    MC(MPI_Bcast(a, mpi_count((size_t)mm * kk), MPI_INT32_T, ar, rowcomm));
    MC(MPI_Bcast(b, mpi_count((size_t)kk * nn), MPI_INT32_T, br, colcomm));
    gemm(a, b, tile, mm, kk, nn, kk, nn, nn, step != 0, NULL);
  }
  int *sc = mem_new(world_size, 4, 1), *sd = mem_new(world_size, 4, 0),
      *rc = mem_new(world_size, 4, 1), *rd = mem_new(world_size, 4, 0);
  int total = 0;
  for (int r = 0; r < world_size; r++) {
    int x = part_start(opt.m, world_size, r), y = x + part_size(opt.m, world_size, r);
    if (x < mi)
      x = mi;
    if (y > mi + mm)
      y = mi + mm;
    sc[r] = x < y ? mpi_count((size_t)(y - x) * nn) : 0;
    sd[r] = total;
    total += sc[r];
  }
  int64_t *send = mem_new(total, 8, 0);
  for (int r = 0; r < world_size; r++) {
    int x = part_start(opt.m, world_size, r);
    if (x < mi)
      x = mi;
    if (sc[r])
      memcpy(send + sd[r], tile + (size_t)(x - mi) * nn, (size_t)sc[r] * 8);
  }
  MC(MPI_Alltoall(sc, 1, MPI_INT, rc, 1, MPI_INT, MPI_COMM_WORLD));
  int receive = 0;
  for (int r = 0; r < world_size; r++) {
    rd[r] = receive;
    receive += rc[r];
  }
  int64_t *recv = mem_new(receive, 8, 0);
  MC(MPI_Alltoallv(send, sc, sd, MPI_INT64_T, recv, rc, rd, MPI_INT64_T,
                   MPI_COMM_WORLD));
  alloc_result(in, d, rows, r0, ld);
  for (int r = 0; r < world_size; r++) {
    int gr = r / pc, gc = r % pc, begin = part_start(opt.m, pr, gr),
        end = begin + part_size(opt.m, pr, gr), col = part_start(opt.n, pc, gc),
        width = part_size(opt.n, pc, gc);
    if (begin < *r0)
      begin = *r0;
    if (end > *r0 + *rows)
      end = *r0 + *rows;
    size_t off = (size_t)rd[r];
    for (int i = begin; i < end; i++) {
      memcpy(*d + (size_t)(i - *r0) * *ld + col, recv + off, (size_t)width * 8);
      off += width;
    }
  }
  add_c(in, *r0, *rows, *d, *ld);
  mem_drop(tile);
  mem_drop(a);
  mem_drop(b);
  mem_drop(sc);
  mem_drop(sd);
  mem_drop(rc);
  mem_drop(rd);
  mem_drop(send);
  mem_drop(recv);
  MC(MPI_Comm_free(&rowcomm));
  MC(MPI_Comm_free(&colcomm));
}
