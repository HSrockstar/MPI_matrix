/*
 * mpi_matmul.c — 基于 MPI 的 D = A*B + C（纯消息传递，无共享内存并行）
 * 版本 3：在 v2（I/O 并行化）基础上增加 kernel 两级 cache 分块（TK×TI），
 *          解决 4 进程时 4 份私有 B（33.6MB）逼近 36MB L3 造成的相互驱逐。
 *
 * 矩阵规模（固定于作业输入）：
 *   A: M×K = 1023×2047   B: K×N = 2047×1023   C/D: M×N = 1023×1023
 *   文本格式固定 4096 行：#A | 1023 行 A | #B | 2047 行 B | #C | 1023 行 C
 *
 * v2 相对 v1 的改动（程序任何阶段都不再依赖单一进程）：
 *   读入  1) 每个进程对输入文件做只读私有 mmap（各进程地址空间独立，
 *            语义上是文件 I/O，不是进程间共享内存）
 *         2) 各进程 memchr 扫描自己负责的字节片内的换行符，
 *            MPI_Allgatherv 汇成全局"行号 → 文件偏移"表
 *         3) 各进程只解析自己负责的 A/C 行块；B 按行块分片解析，
 *            直接解析进 B + 本进程偏移处，再用就地 MPI_Allgatherv 拼整
 *         → v1 中进程 0 读全文件 + Scatter A/C + Bcast B 的通信全部消失
 *   写出  4) 各进程把自己算出的 D 行格式化成文本块，MPI_Allgather 块长
 *            得到各块起始偏移，pwrite 各写各的区间（进程 0 预创建并清空
 *            文件，屏障保证清空先于任何写入）→ 不再 Gather 到进程 0
 *
 * 构建: mpicc -O3 -Wall -march=native -o mpi_matmul mpi_matmul.c
 * 运行: mpirun -np <p> ./mpi_matmul <RandMatrix.txt> <result.txt>
 */
#include <fcntl.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define M 1023 /* A、C、D 的行数 */
#define K 2047 /* A 的列数 = B 的行数 */
#define N 1023 /* B、C、D 的列数 */

/* kernel 分块参数（可被 -DTK=.. -DTI=.. 覆盖）：
 * TK —— k 方向条带宽度：B 的一个条带 TK×N 个 int 驻留 L2（约 TK*4KB），
 *        避免每算一行 D 都要从 L3/内存重新流过整份 B（4 进程时 4 份私有 B
 *        共 33.6MB，逼近 36MB L3，不分块会互相驱逐）
 * TI —— i 方向行块。扫描实验（TK∈{128,256,512}×TI∈{1,2,4,8}）表明
 *        本机上 TI=1 最优（GCC 对单行 axpy 内层的寄存器分配更好，
 *        i 方向展开反而变慢），收益全部来自 k 条带驻留 L2 */
#ifndef TK
#define TK 128 /* 128×1023×4B ≈ 512KB，稳驻 2MB P 核 L2 */
#endif
#ifndef TI
#define TI 1
#endif

/* 输入文件固定 4096 行（行号从 0 起）：#A 第 0 行，#B 第 1024 行，#C 第 3072 行 */
#define TOTAL_LINES 4096
#define LINE_MARK_A 0
#define LINE_MARK_B 1024
#define LINE_MARK_C 3072
#define LINE_A0 1    /* A 的第 0 行所在行号 */
#define LINE_B0 1025 /* B 的第 0 行所在行号 */
#define LINE_C0 3073 /* C 的第 0 行所在行号 */

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

/* 只读私有 mmap 整个文件，返回映射首地址与文件长度 */
static const char *map_file(const char *path, size_t *out_len)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0)
        die("fstat input file failed");
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED)
        die("mmap input file failed");
    close(fd);
    *out_len = (size_t)st.st_size;
    return (const char *)p;
}

/* 从 [p, end) 解析 cnt 个整数写入 out，返回停止位置；'#'-标记与换行自动跳过 */
static const char *parse_ints(const char *p, const char *end, int *out, size_t cnt)
{
    for (size_t c = 0; c < cnt; c++) {
        while (p < end && *p != '-' && (*p < '0' || *p > '9'))
            p++;
        if (p >= end)
            die("input file ended earlier than expected");
        int neg = 0, v = 0;
        if (*p == '-') {
            neg = 1;
            p++;
        }
        while (p < end && *p >= '0' && *p <= '9') {
            v = v * 10 + (*p - '0');
            p++;
        }
        out[c] = neg ? -v : v;
    }
    return p;
}

/* 收集 buf[lo,hi) 内所有 '\n' 的文件偏移（memchr 为向量化扫描） */
static size_t scan_newlines(const char *buf, size_t lo, size_t hi, uint32_t **out)
{
    size_t cap = 1024, n = 0;
    uint32_t *v = xmalloc(cap * sizeof *v);
    const char *p = buf + lo, *end = buf + hi;
    while (p < end) {
        const char *nl = memchr(p, '\n', (size_t)(end - p));
        if (!nl)
            break;
        if (n == cap) {
            cap *= 2;
            v = realloc(v, cap * sizeof *v);
            if (!v)
                die("out of memory");
        }
        v[n++] = (uint32_t)(nl - buf);
        p = nl + 1;
    }
    *out = v;
    return n;
}

/* total 行按行块划分，生成集合通信用的 counts/displacements（stride 为行宽） */
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

int main(int argc, char **argv)
{
    const char *in_path = argc > 1 ? argv[1] : "RandMatrix.txt";
    const char *out_path = argc > 2 ? argv[2] : "result.txt";

    MPI_Init(&argc, &argv);
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    double t_start = MPI_Wtime();

    /* M=1023、K=2047 都不能被 2/4 整除：行块划分带余数，靠不均匀 counts 处理 */
    int base = M / nproc, rem = M % nproc;
    int my_rows = base + (rank < rem);
    int my_first = rank * base + (rank < rem ? rank : rem); /* A/C 行块 */

    int base_b = K / nproc, rem_b = K % nproc;
    int my_rows_b = base_b + (rank < rem_b);
    int my_first_b = rank * base_b + (rank < rem_b ? rank : rem_b); /* B 行片 */

    int *B = xmalloc((size_t)K * N * sizeof *B);
    int *Al = xmalloc((size_t)my_rows * K * sizeof *Al);
    int *Cl = xmalloc((size_t)my_rows * N * sizeof *Cl);
    int *Dl = xmalloc((size_t)my_rows * N * sizeof *Dl);
    int *cnt_bn = xmalloc((size_t)nproc * sizeof *cnt_bn); /* B 行片，stride N */
    int *dsp_bn = xmalloc((size_t)nproc * sizeof *dsp_bn);
    block_layout(nproc, K, N, cnt_bn, dsp_bn);

    /* ---- 1a. mmap 输入；各进程扫描自己字节片内的换行符 ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double r0 = MPI_Wtime();
    size_t fsize;
    const char *buf = map_file(in_path, &fsize);
    size_t lo = fsize * (size_t)rank / (size_t)nproc;
    size_t hi = fsize * (size_t)(rank + 1) / (size_t)nproc;
    uint32_t *nl_loc;
    size_t nl_cnt = scan_newlines(buf, lo, hi, &nl_loc);
    double t_scan = MPI_Wtime() - r0;

    /* ---- 1b. Allgatherv 汇成全局换行表 → 每进程得到 line_off[行号] ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    double c0 = MPI_Wtime();
    int *ncnt = xmalloc((size_t)nproc * sizeof *ncnt);
    int *ndsp = xmalloc((size_t)nproc * sizeof *ndsp);
    int nl_ci = (int)nl_cnt;
    MPI_Allgather(&nl_ci, 1, MPI_INT, ncnt, 1, MPI_INT, MPI_COMM_WORLD);
    int nl_total = 0;
    for (int q = 0; q < nproc; q++) {
        ndsp[q] = nl_total;
        nl_total += ncnt[q];
    }
    uint32_t *nl_all = xmalloc((size_t)nl_total * sizeof *nl_all);
    MPI_Allgatherv(nl_loc, nl_ci, MPI_UINT32_T, nl_all, ncnt, ndsp, MPI_UINT32_T,
                   MPI_COMM_WORLD);
    if (nl_total != TOTAL_LINES)
        die("输入文件行数与作业格式不符（应为 4096 行）");
    uint32_t *line_off = xmalloc((TOTAL_LINES + 1) * sizeof *line_off);
    line_off[0] = 0;
    for (int i = 0; i < TOTAL_LINES; i++)
        line_off[i + 1] = nl_all[i] + 1;
    if (memcmp(buf + line_off[LINE_MARK_A], "#A", 2) != 0 ||
        memcmp(buf + line_off[LINE_MARK_B], "#B", 2) != 0 ||
        memcmp(buf + line_off[LINE_MARK_C], "#C", 2) != 0)
        die("未在预期行号找到 #A/#B/#C 标记");
    double t_table = MPI_Wtime() - c0;

    /* ---- 1c. 各进程只解析自己的 A/C 行块与 B 行片（B 直接进自己偏移处） ---- */
    double p0 = MPI_Wtime();
    parse_ints(buf + line_off[LINE_A0 + my_first], buf + fsize, Al,
               (size_t)my_rows * K);
    parse_ints(buf + line_off[LINE_B0 + my_first_b], buf + fsize,
               B + dsp_bn[rank], (size_t)my_rows_b * N);
    parse_ints(buf + line_off[LINE_C0 + my_first], buf + fsize, Cl,
               (size_t)my_rows * N);
    double t_parse = MPI_Wtime() - p0;
    double t_read = t_scan + t_parse;

    /* ---- 1d. 就地 Allgatherv：各进程贡献已在 B + dsp_bn[rank] 处，拼成完整 B ---- */
    double c1 = MPI_Wtime();
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_INT, B, cnt_bn, dsp_bn, MPI_INT,
                   MPI_COMM_WORLD);
    double t_comm = t_table + (MPI_Wtime() - c1);

    /* ---- 2. 本地计算 Dl = Al × B + Cl：k 条带 × i 行块 两级分块 ----
     * 循环序 kk → i0 → k → i → j（j 为最内层，自动向量化）：
     *   B 的 TK 行条带在 i0 行块间驻留 L2；条带内每个 B 行进入 L1 后
     *   被 TI 个 D 行复用；restrict 告知编译器三个矩阵互不重叠 */
    double k0 = MPI_Wtime();
    {
        int *restrict const d = Dl;
        const int *restrict const am = Al;
        const int *restrict const bm = B;
        for (int i = 0; i < my_rows; i++) /* D 初值 = C */
            memcpy(d + (size_t)i * N, Cl + (size_t)i * N, (size_t)N * sizeof *d);
        for (int kk = 0; kk < K; kk += TK) {
            const int kend = (kk + TK <= K) ? kk + TK : K;
            for (int i0 = 0; i0 < my_rows; i0 += TI) {
                const int iend = (i0 + TI <= my_rows) ? i0 + TI : my_rows;
                for (int k = kk; k < kend; k++) {
                    const int *brow = bm + (size_t)k * N;
                    for (int i = i0; i < iend; i++) {
                        const int av = am[(size_t)i * K + k];
                        int *drow = d + (size_t)i * N;
                        for (int j = 0; j < N; j++)
                            drow[j] += av * brow[j];
                    }
                }
            }
        }
    }
    double t_kernel = MPI_Wtime() - k0;

    /* ---- 3. 各进程格式化自己的 D 行块（进程 0 的块含 "#D\n" 头） ---- */
    double w0 = MPI_Wtime();
    char *obuf = xmalloc((size_t)my_rows * N * 12 + 16 + (rank == 0 ? 3 : 0));
    size_t op = 0;
    if (rank == 0) {
        memcpy(obuf, "#D\n", 3);
        op = 3;
    }
    for (int i = 0; i < my_rows; i++) {
        const int *row = Dl + (size_t)i * N;
        for (int j = 0; j < N; j++) {
            op += put_int(obuf + op, row[j]);
            obuf[op++] = (j == N - 1) ? '\n' : ' ';
        }
    }
    double t_fmt = MPI_Wtime() - w0;

    /* ---- 4. Allgather 块长 → pwrite 各写各的字节区间 ---- */
    if (rank == 0) { /* 预创建并清空结果文件；下面的屏障保证清空先于任何写入 */
        int tfd = open(out_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (tfd < 0)
            die("cannot create result file");
        close(tfd);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double i0 = MPI_Wtime();
    int my_blk = (int)op;
    int *blk_cnt = xmalloc((size_t)nproc * sizeof *blk_cnt);
    int *blk_dsp = xmalloc((size_t)nproc * sizeof *blk_dsp);
    MPI_Allgather(&my_blk, 1, MPI_INT, blk_cnt, 1, MPI_INT, MPI_COMM_WORLD);
    blk_dsp[0] = 0;
    for (int q = 1; q < nproc; q++)
        blk_dsp[q] = blk_dsp[q - 1] + blk_cnt[q - 1];
    int ofd = open(out_path, O_WRONLY);
    if (ofd < 0)
        die("cannot open result file for writing");
    if (pwrite(ofd, obuf, op, (off_t)blk_dsp[rank]) != (ssize_t)op)
        die("pwrite result file failed");
    close(ofd);
    double t_write = t_fmt + (MPI_Wtime() - i0);

    /* ---- 计时汇总（whole-program 口径：各进程取最大值） ---- */
    double t_total = MPI_Wtime() - t_start;
    double tmax[5] = {t_read, t_comm, t_kernel, t_write, t_total};
    double tred[5];
    MPI_Reduce(tmax, tred, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("np=%d  read=%.3f  comm=%.3f  kernel(max)=%.3f  write=%.3f  TOTAL(max)=%.3f s\n",
               nproc, tred[0], tred[1], tred[2], tred[3], tred[4]);
    fflush(stdout);

    munmap((void *)buf, fsize);
    free(nl_loc);
    free(nl_all);
    free(line_off);
    free(ncnt);
    free(ndsp);
    free(obuf);
    free(blk_cnt);
    free(blk_dsp);
    free(B);
    free(Al);
    free(Cl);
    free(Dl);
    free(cnt_bn);
    free(dsp_bn);

    MPI_Finalize();
    return 0;
}
