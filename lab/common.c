#include "lab.h"
#include <limits.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

Options opt;
int rank_id, world_size;
typedef struct Block {
  void *base;
  size_t bytes;
  int mapped;
  struct Block *next;
} Block;
static Block *blocks;
static unsigned char *arena_ptr;
static size_t arena_left;
typedef struct ArenaItem {
  void *base;
  size_t bytes;
  int busy;
  struct ArenaItem *next;
} ArenaItem;
static ArenaItem *arena_items;

void fail(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "rank %d: ", rank_id);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  va_end(ap);
  MPI_Abort(MPI_COMM_WORLD, 2);
  abort();
}
void mpi_check(int code, const char *op) {
  if (code != MPI_SUCCESS) {
    char s[MPI_MAX_ERROR_STRING];
    int n = 0;
    MPI_Error_string(code, s, &n);
    fail("%s: %.*s", op, n, s);
  }
}
size_t checked_mul(size_t a, size_t b) {
  if (b && a > SIZE_MAX / b)
    fail("size overflow");
  return a * b;
}
int mpi_count(size_t n) {
  if (n > INT_MAX)
    fail("MPI count exceeds INT_MAX");
  return (int)n;
}
int part_size(int n, int p, int r) { return n / p + (r < n % p); }
int part_start(int n, int p, int r) { return r * (n / p) + (r < n % p ? r : n % p); }
static void *allocate(size_t n) {
  size_t alignment = !strcmp(opt.alloc, "huge") ? 2097152 : 64;
  if (n > SIZE_MAX - alignment)
    fail("allocation overflow");
  size_t bytes = (n + alignment - 1) & ~(alignment - 1);
  if (!bytes)
    bytes = alignment;
  void *base = NULL;
  int mapped = 0;
  if (!strcmp(opt.alloc, "huge")) {
    if (bytes > SIZE_MAX - alignment)
      fail("huge mapping size overflow");
    size_t full = bytes + alignment;
    unsigned char *raw =
        mmap(NULL, full, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw != MAP_FAILED) {
      uintptr_t addr = ((uintptr_t)raw + alignment - 1) & ~(uintptr_t)(alignment - 1);
      base = (void *)addr;
      size_t lead = addr - (uintptr_t)raw;
      if (lead)
        munmap(raw, lead);
      size_t tail = full - lead - bytes;
      if (tail)
        munmap((char *)base + bytes, tail);
      mapped = 1;
#ifdef MADV_HUGEPAGE
      if (madvise(base, bytes, MADV_HUGEPAGE) && rank_id == 0)
        fprintf(stderr, "THP advice unavailable; ordinary pages retained\n");
#endif
    }
  }
  if (!base && posix_memalign(&base, alignment, bytes))
    fail("allocation failed (%zu bytes)", bytes);
  Block *q = malloc(sizeof *q);
  if (!q)
    fail("allocation metadata failed");
  *q = (Block){base, bytes, mapped, blocks};
  blocks = q;
  return base;
}
void *mem_new(size_t n, size_t width, int zero) {
  size_t bytes = checked_mul(n, width);
  void *p;
  if (opt.arena) {
    if (bytes > SIZE_MAX - 63)
      fail("arena size overflow");
    size_t step = (bytes + 63) & ~(size_t)63;
    if (!step)
      step = 64;
    for (ArenaItem *q = arena_items; q; q = q->next) {
      if (!q->busy && q->bytes >= step) {
        q->busy = 1;
        if (zero)
          memset(q->base, 0, bytes);
        return q->base;
      }
    }
    if (step > arena_left) {
      size_t chunk = step > 8 * 1024 * 1024 ? step : 8 * 1024 * 1024;
      arena_ptr = allocate(chunk);
      arena_left = chunk;
    }
    p = arena_ptr;
    arena_ptr += step;
    arena_left -= step;
    ArenaItem *q = malloc(sizeof *q);
    if (!q)
      fail("arena metadata allocation failed");
    *q = (ArenaItem){p, step, 1, arena_items};
    arena_items = q;
  } else
    p = allocate(bytes);
  if (zero)
    memset(p, 0, bytes);
  return p;
}
void mem_drop(void *p) {
  if (!p)
    return;
  if (opt.arena) {
    for (ArenaItem *q = arena_items; q; q = q->next)
      if (q->base == p) {
        if (!q->busy)
          fail("arena double release");
        q->busy = 0;
        return;
      }
    fail("invalid arena release");
  }
  Block **it = &blocks;
  while (*it && (*it)->base != p)
    it = &(*it)->next;
  if (!*it)
    fail("invalid private allocation release");
  Block *q = *it;
  *it = q->next;
  if (q->mapped)
    munmap(q->base, q->bytes);
  else
    free(q->base);
  free(q);
}
void mem_finish(void) {
  while (arena_items) {
    ArenaItem *q = arena_items;
    arena_items = q->next;
    free(q);
  }
  while (blocks) {
    Block *q = blocks;
    blocks = q->next;
    if (q->mapped)
      munmap(q->base, q->bytes);
    else
      free(q->base);
    free(q);
  }
  arena_ptr = NULL;
  arena_left = 0;
}
