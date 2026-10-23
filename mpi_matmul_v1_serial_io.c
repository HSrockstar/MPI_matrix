/*
 * mpi_matmul.c — 基于 MPI 的 D = A*B + C（行块划分，纯消息传递，无共享内存并行）
 *
 * 矩阵规模（固定于作业输入）：
 *   A: M×K = 1023×2047   B: K×N = 2047×1023   C/D: M×N = 1023×1023
 *
 * 并行方案（框架版）：
 *   1. 进程 0 读取整个 RandMatrix.txt（快读 + 手写整数解析）
 *   2. MPI_Scatterv 按行块分发 A、C 的各行，MPI_Bcast 广播整个 B
 *   3. 各进程本地计算 D_local = A_local × B + C_local（i-k-j 循环序，B 的行连续访问）
 *   4. MPI_Gatherv 收集 D 到进程 0，手写快速格式化输出 result.txt
 *
 * 计时（作业要求：整个程序的运行时间）：
 *   total = MPI_Init 之后到写完 result.txt 的墙上时间（取所有进程最大值），
 *   并分阶段打印 read / comm / kernel / write，便于后续优化分析。
 *
 * 构建: mpicc -O3 -Wall -march=native -o mpi_matmul mpi_matmul.c
 * 运行: mpirun -np <p> ./mpi_matmul <RandMatrix.txt> <result.txt>
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M 1023 /* A、C、D 的行数 */
#define K 2047 /* A 的列数 = B 的行数 */
#define N 1023 /* B、C、D 的列数 */

static void die(const char *msg)
{
    fprintf(stderr, "error: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p)
        die("out of memory");
    return p;
}

/* 一次性读入整个文件，返回 NUL 结尾缓冲区 */
static char *read_file(const char *path, size_t *out_len)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = xmalloc((size_t)sz + 1);
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz)
        die("failed to read input file");
    fclose(f);
    buf[sz] = '\0';
    *out_len = (size_t)sz;
    return buf;
}

/* 从 buf[*pos] 起解析 cnt 个整数写入 out；自动跳过 #A/#B/#C 标记行 */
static void parse_ints(const char *buf, size_t len, size_t *pos, int *out, size_t cnt)
{
    size_t p = *pos;
    for (size_t c = 0; c < cnt; c++) {
        while (p < len && buf[p] != '-' && (buf[p] < '0' || buf[p] > '9'))
            p++;
        if (p >= len)
            die("input file ended earlier than expected");
        int neg = 0, v = 0;
        if (buf[p] == '-') {
            neg = 1;
            p++;
        }
        while (p < len && buf[p] >= '0' && buf[p] <= '9') {
            v = v * 10 + (buf[p] - '0');
            p++;
        }
        out[c] = neg ? -v : v;
    }
    *pos = p;
}

/* 十进制写入 v 到 dst，返回字符数 */
static size_t put_int(char *dst, int v)
{
    size_t n = 0;
    unsigned long long u;
    if (v < 0) {
        dst[n++] = '-';
        u = 0ULL - (unsigned long long)v;
    } else {
        u = (unsigned long long)v;
    }
    char tmp[20];
    int t = 0;
    do {
        tmp[t++] = (char)('0' + u % 10);
        u /= 10;
    } while (u);
    while (t)
        dst[n++] = tmp[--t];
    return n;
}

/* total 行按行块划分，生成 Scatterv/Gatherv 的 counts/displacements（stride 为行宽） */
static void block_layout(int nproc, int total, int stride, int *cnt, int *dsp)
{
    int base = total / nproc, rem = total % nproc, off = 0;
    for (int q = 0; q < nproc; q++) {
        int rows = base + (q < rem);
        cnt[q] = rows * stride;
        dsp[q] = off;
        off += cnt[q];
    }
}

int main(int argc, char **argv)
{
    const char *in_path = argc > 1 ? argv[1] : "RandMatrix.txt";
    const char *out_path = argc > 2 ? argv[2] : "result.txt";

    MPI_Init(&argc, &argv);
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    double t_start = MPI_Wtime();

    /* M=1023 不能被 2/4 整除：行块划分带余数，靠 Scatterv 的不均匀 counts 处理 */
    const int base = M / nproc, rem = M % nproc;
    const int my_rows = base + (rank < rem);

    int *B = xmalloc((size_t)K * N * sizeof *B); /* 每进程一份完整 B（由广播得到） */
    int *Al = xmalloc((size_t)my_rows * K * sizeof *Al);
    int *Cl = xmalloc((size_t)my_rows * N * sizeof *Cl);
    int *Dl = xmalloc((size_t)my_rows * N * sizeof *Dl);

    int *A = NULL, *C = NULL, *D = NULL; /* 完整矩阵，仅进程 0 持有 */
    int *cnt_k = xmalloc((size_t)nproc * sizeof *cnt_k);
    int *dsp_k = xmalloc((size_t)nproc * sizeof *dsp_k);
    int *cnt_n = xmalloc((size_t)nproc * sizeof *cnt_n);
    int *dsp_n = xmalloc((size_t)nproc * sizeof *dsp_n);
    block_layout(nproc, M, K, cnt_k, dsp_k);
    block_layout(nproc, M, N, cnt_n, dsp_n);

    /* ---- 1. 进程 0 读入 A、B、C ---- */
    double t_read = 0.0;
    if (rank == 0) {
        double r0 = MPI_Wtime();
        size_t len, pos = 0;
        char *buf = read_file(in_path, &len);
        A = xmalloc((size_t)M * K * sizeof *A);
        C = xmalloc((size_t)M * N * sizeof *C);
        parse_ints(buf, len, &pos, A, (size_t)M * K);
        parse_ints(buf, len, &pos, B, (size_t)K * N);
        parse_ints(buf, len, &pos, C, (size_t)M * N);
        free(buf);
        t_read = MPI_Wtime() - r0;
    }

    /* ---- 2. 分发：A、C 按行块 Scatter，B 整体 Bcast ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double c0 = MPI_Wtime();
    MPI_Scatterv(A, cnt_k, dsp_k, MPI_INT, Al, my_rows * K, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(C, cnt_n, dsp_n, MPI_INT, Cl, my_rows * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, K * N, MPI_INT, 0, MPI_COMM_WORLD);
    double t_comm = MPI_Wtime() - c0;

    /* ---- 3. 本地计算 Dl = Al × B + Cl（i-k-j 序：B 按行连续访问，缓存友好） ---- */
    double k0 = MPI_Wtime();
    for (int i = 0; i < my_rows; i++) {
        const int *arow = Al + (size_t)i * K;
        int *drow = Dl + (size_t)i * N;
        memcpy(drow, Cl + (size_t)i * N, (size_t)N * sizeof *drow); /* D 初值 = C */
        for (int k = 0; k < K; k++) {
            const int a = arow[k];
            const int *brow = B + (size_t)k * N;
            for (int j = 0; j < N; j++)
                drow[j] += a * brow[j];
        }
    }
    double t_kernel = MPI_Wtime() - k0;

    /* ---- 4. 收集 D 到进程 0 并写出 result.txt ---- */
    if (rank == 0)
        D = xmalloc((size_t)M * N * sizeof *D);
    MPI_Gatherv(Dl, my_rows * N, MPI_INT, D, cnt_n, dsp_n, MPI_INT, 0, MPI_COMM_WORLD);

    double t_write = 0.0;
    if (rank == 0) {
        double w0 = MPI_Wtime();
        FILE *f = fopen(out_path, "w");
        if (!f)
            die("cannot open result file for writing");
        fputs("#D\n", f);
        char *obuf = xmalloc(((size_t)M * N) * 12 + 16);
        size_t op = 0;
        for (int i = 0; i < M; i++) {
            const int *row = D + (size_t)i * N;
            for (int j = 0; j < N; j++) {
                op += put_int(obuf + op, row[j]);
                obuf[op++] = (j == N - 1) ? '\n' : ' ';
            }
        }
        fwrite(obuf, 1, op, f);
        fclose(f);
        free(obuf);
        t_write = MPI_Wtime() - w0;
    }

    /* ---- 计时汇总（whole-program 口径：MPI_Wtime total 取各进程最大值） ---- */
    double t_total = MPI_Wtime() - t_start;
    double t_kernel_max, t_total_max;
    MPI_Reduce(&t_kernel, &t_kernel_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_total, &t_total_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("np=%d  read=%.3f  comm=%.3f  kernel(max)=%.3f  write=%.3f  TOTAL(max)=%.3f s\n",
               nproc, t_read, t_comm, t_kernel_max, t_write, t_total_max);
    fflush(stdout);

    free(B); free(Al); free(Cl); free(Dl);
    free(A); free(C); free(D);
    free(cnt_k); free(dsp_k); free(cnt_n); free(dsp_n);

    MPI_Finalize();
    return 0;
}
