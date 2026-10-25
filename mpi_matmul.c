#include "lab/lab.h"
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
static int integer(const char *s) {
  char *end;
  errno = 0;
  long n = strtol(s, &end, 10);
  if (errno || end == s || *end || n < 0 || n > 1000000)
    fail("invalid numeric option: %s", s);
  return (int)n;
}
static int oneof(const char *s, const char *list) {
  size_t n = strlen(s);
  const char *p = list;
  while (*p) {
    const char *end = strchr(p, ',');
    if (!end)
      end = p + strlen(p);
    if ((size_t)(end - p) == n && !memcmp(p, s, n))
      return 1;
    p = *end ? end + 1 : end;
  }
  return 0;
}
static void choice(const char *s, const char *list, const char *name) {
  if (!oneof(s, list))
    fail("unknown %s: %s (choices %s)", name, s, list);
}
static void defaults(void) {
  opt = (Options){.m = 1023,
                  .k = 2047,
                  .n = 1023,
                  .kb = 2048,
                  .mr = 6,
                  .nr = 16,
                  .rows = 96,
                  .pad = 0,
                  .unroll = 1,
                  .progress = 1,
                  .leaf = 32,
                  .kernel = "auto",
                  .dist = "rows",
                  .bmode = "allgatherv",
                  .parser = "scalar",
                  .reader = "mmap",
                  .prefault = "none",
                  .format = "pairs",
                  .writer = "pwrite",
                  .offset = "exscan",
                  .alloc = "aligned",
                  .layout = "rows",
                  .input = "RandMatrix.txt",
                  .output = "result.txt",
                  .beta0 = 1};
}
static void options(int argc, char **argv) {
  int positional = 0;
  for (int i = 1; i < argc; i++) {
    const char *s = argv[i];
    if (!strcmp(s, "--features")) {
      if (!rank_id)
        printf("{\"avx2\":%d,\"avxvnni\":%d}\n", has_avx2(), has_vnni());
      MPI_Finalize();
      exit(0);
    }
    if (!strcmp(s, "--help")) {
      if (!rank_id)
        puts("mpi_matmul [input output] [--m M --k K --n N] [--kernel "
             "auto|vnni|avx2-sign|nibble|scalar|fp32|fp64|winograd|recursive|"
             "blas|blas-fp32|external|strassen|strassen-winograd] "
             "[--dist rows|ring|split-k|summa|columns] [--bmode "
             "allgatherv|equal|packed|root|root-packed|replicate] [--kb K --mr "
             "R --nr N "
             "--rows R --pad BYTES --unroll 1|2|4 --pipeline 0|1 --leaf N] "
             "[--parser scalar|simd|swar|wide "
             "--reader mmap|pread --prefault none|local|all] [--format "
             "pairs|quad|jeaiii|snprintf|batch|vector --writer "
             "pwrite|mpi|mpi-all "
             "--offset exscan|allgather] [--alloc aligned|huge --arena 0|1 "
             "--layout rows|interleaved --beta0 0|1 --progress 0|1] "
             "[--stream 0|1 --fused 0|1 --static-table 0|1 --balance WEIGHT "
             "--write-chunk BYTES] "
             "[--aligned 0|1 --progress-us N --memory-stats 0|1] "
             "[--selftest|--features]");
      MPI_Finalize();
      exit(0);
    }
    if (!strcmp(s, "--selftest")) {
      selftest();
      MPI_Finalize();
      exit(0);
    }
    if (strncmp(s, "--", 2)) {
      if (positional == 0)
        opt.input = s;
      else if (positional == 1)
        opt.output = s;
      else
        fail("extra positional argument");
      positional++;
      continue;
    }
    if (i + 1 == argc)
      fail("missing value for %s", s);
    const char *v = argv[++i];
    struct {
      const char *name;
      int *value;
    } numbers[] = {{"--aligned", &opt.aligned},
                   {"--progress-us", &opt.progress_us},
                   {"--memory-stats", &opt.memory_stats},
                   {"--m", &opt.m},
                   {"--k", &opt.k},
                   {"--n", &opt.n},
                   {"--kb", &opt.kb},
                   {"--mr", &opt.mr},
                   {"--nr", &opt.nr},
                   {"--rows", &opt.rows},
                   {"--pad", &opt.pad},
                   {"--unroll", &opt.unroll},
                   {"--pipeline", &opt.pipeline},
                   {"--leaf", &opt.leaf},
                   {"--progress", &opt.progress},
                   {"--beta0", &opt.beta0},
                   {"--arena", &opt.arena},
                   {"--stream", &opt.stream},
                   {"--fused", &opt.fused},
                   {"--static-table", &opt.static_table},
                   {"--balance", &opt.balance}};
    int found = 0;
    for (size_t j = 0; j < sizeof numbers / sizeof *numbers; j++)
      if (!strcmp(s, numbers[j].name)) {
        *numbers[j].value = integer(v);
        found = 1;
        break;
      }
    if (found)
      continue;
    if (!strcmp(s, "--write-chunk")) {
      opt.write_chunk = (size_t)integer(v);
      continue;
    }
    struct {
      const char *name;
      const char **value;
    } strings[] = {{"--kernel", &opt.kernel}, {"--dist", &opt.dist},
                   {"--bmode", &opt.bmode},   {"--parser", &opt.parser},
                   {"--reader", &opt.reader}, {"--prefault", &opt.prefault},
                   {"--format", &opt.format}, {"--writer", &opt.writer},
                   {"--offset", &opt.offset}, {"--alloc", &opt.alloc},
                   {"--layout", &opt.layout}};
    for (size_t j = 0; j < sizeof strings / sizeof *strings; j++)
      if (!strcmp(s, strings[j].name)) {
        *strings[j].value = v;
        found = 1;
        break;
      }
    if (found)
      continue;
    fail("unknown option %s", s);
  }
  if (!opt.m || !opt.k || !opt.n || !opt.kb || !opt.rows || !opt.leaf)
    fail("dimensions/block sizes must be positive");
  if (!((opt.mr == 4 && opt.nr == 16) || (opt.mr == 6 && opt.nr == 16) ||
        (opt.mr == 8 && opt.nr == 8) || (opt.mr == 4 && opt.nr == 24)))
    fail("tile must be 4x16, 6x16, 8x8, or 4x24");
  if (opt.rows % opt.mr || opt.pad % 64)
    fail("rows must be multiple of MR; pad must be multiple of 64 bytes");
  if (opt.unroll != 1 && opt.unroll != 2 && opt.unroll != 4)
    fail("unroll must be 1, 2 or 4");
  if (opt.beta0 > 1 || opt.progress > 1 || opt.arena > 1 || opt.pipeline > 1 ||
      opt.fused > 1 || opt.aligned > 1 || opt.memory_stats > 1)
    fail("boolean options require 0 or 1");
  choice(opt.kernel,
         "auto,vnni,avx2-sign,nibble,scalar,fp32,fp64,winograd,recursive,blas,"
         "blas-fp32,external,strassen,strassen-winograd",
         "kernel");
  choice(opt.dist, "rows,ring,split-k,summa,columns", "distribution");
  choice(opt.bmode, "allgatherv,equal,packed,root,root-packed,replicate", "B mode");
  choice(opt.parser, "scalar,simd,swar,wide", "parser");
  choice(opt.reader, "mmap,pread", "reader");
  choice(opt.prefault, "none,local,all", "prefault");
  choice(opt.format, "pairs,quad,jeaiii,snprintf,batch,vector", "format");
  choice(opt.writer, "pwrite,mpi,mpi-all", "writer");
  choice(opt.offset, "exscan,allgather", "offset");
  choice(opt.alloc, "aligned,huge", "allocator");
  choice(opt.layout, "rows,interleaved", "A layout");
  if (opt.stream > 1 ||
      (opt.stream &&
       (strcmp(opt.dist, "rows") || (!oneof(opt.bmode, "root,root-packed,replicate")))))
    fail("stream requires rows distribution with root/root-packed/replicate B");
  if ((!strcmp(opt.bmode, "root-packed") || opt.fused) && !opt.stream)
    fail("root-packed and fused parsing require stream=1");
  if (opt.static_table > 1)
    fail("static-table requires 0 or 1");
  if (opt.balance && !oneof(opt.dist, "rows,ring"))
    fail("balance is supported for rows/ring only");
  mpi_count((size_t)opt.m * opt.n);
  mpi_count((size_t)opt.k * opt.n);
}
int main(int argc, char **argv) {
  struct timespec wall0;
  clock_gettime(CLOCK_MONOTONIC, &wall0);
  /* The experiment contract is one application thread per MPI rank. */
  if (setenv("OMP_NUM_THREADS", "1", 1) || setenv("OPENBLAS_NUM_THREADS", "1", 1) ||
      setenv("MKL_NUM_THREADS", "1", 1)) {
    perror("setenv");
    return 2;
  }
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  defaults();
  options(argc, argv);
  double start = MPI_Wtime();
  Input in = input_open(opt.input);
  int64_t bounds[3];
  input_bounds(&in, bounds);
  kernel_validate(bounds[0], bounds[1]);
  double read_end = MPI_Wtime();
  int64_t *d = NULL;
  int rows = 0, r0 = 0, ld = 0;
  if (opt.stream)
    run_stream(&in);
  else if (!strcmp(opt.dist, "split-k"))
    run_split(&in, &d, &rows, &r0, &ld);
  else if (!strcmp(opt.dist, "summa") || !strcmp(opt.dist, "columns"))
    run_grid(&in, &d, &rows, &r0, &ld);
  else
    run_rows(&in, &d, &rows, &r0, &ld);
  double compute_end = MPI_Wtime();
  if (!opt.stream)
    output_write(d, rows, opt.n, ld);
  if (opt.memory_stats) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage))
      fail("getrusage failed");
    long huge = -1;
    FILE *f = fopen("/proc/self/smaps_rollup", "r");
    if (f) {
      char line[256];
      while (fgets(line, sizeof line, f))
        if (sscanf(line, "AnonHugePages: %ld kB", &huge) == 1)
          break;
      fclose(f);
    }
    fprintf(stderr,
            "{\"rank\":%d,\"maxrss_kb\":%ld,\"minor_faults\":%ld,\"major_"
            "faults\":%ld,\"anon_huge_kb_before_cleanup\":%ld}\n",
            rank_id, usage.ru_maxrss, usage.ru_minflt, usage.ru_majflt, huge);
  }
  input_close(&in);
  mem_finish();
  double end = MPI_Wtime();
  double local[4] = {read_end - start, compute_end - read_end, end - compute_end,
                     end - start},
         maxima[4];
  MC(MPI_Reduce(local, maxima, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
  if (!rank_id)
    printf("{\"np\":%d,\"m\":%d,\"k\":%d,\"n\":%d,\"kernel\":\"%s\",\"dist\":\"%"
           "s\",\"validation_s\":%.9f,\"load_compute_s\":%.9f,\"output_cleanup_"
           "s\":%.9f,\"internal_total_s\":%.9f,\"stream_output_in_compute\":%d,"
           "\"avx2\":%d,\"avxvnni\":%d}\n",
           world_size, opt.m, opt.k, opt.n, opt.kernel, opt.dist, maxima[0], maxima[1],
           maxima[2], maxima[3], opt.stream, has_avx2(), has_vnni());
  MPI_Finalize();
  struct timespec wall1;
  clock_gettime(CLOCK_MONOTONIC, &wall1);
  if (!rank_id)
    fprintf(stderr, "rank0_process_wall_s=%.9f\n",
            (double)(wall1.tv_sec - wall0.tv_sec) +
                (double)(wall1.tv_nsec - wall0.tv_nsec) * 1e-9);
  return 0;
}
