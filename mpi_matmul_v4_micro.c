/*
 * mpi_matmul.c — 基于 MPI 的 D = A*B + C（纯消息传递，无共享内存并行）
 * 版本 4：在 v2（I/O 并行化）+ v3（kernel k 条带分块）基础上，
 *          将 kernel 升级为 BLIS 式"打包 + 寄存器微核"。
 *
 * 矩阵规模（固定于作业输入）：
 *   A: M×K = 1023×2047   B: K×N = 2047×1023   C/D: M×N = 1023×1023
 *   文本格式固定 4096 行：#A | 1023 行 A | #B | 2047 行 B | #C | 1023 行 C
 *
 * v4 kernel 设计（替换 v3 的 k 条带 axpy）：
 *   - pack_b：把 B 重排成 [j块][k][NR 列] 微面板布局（列方向补零到 NPAD），
 *     微核内 k 每步的 NR 个 B 元素完全连续
 *   - micro：一次计算 MR×NR 的 D 块，MR×(NR/8) 个 ymm 累加器跨越整个
 *     K 循环驻留寄存器——axpy 形态"每 8 个乘加回写 1 次 D"变为
 *     "每 MR×NR 个乘加才回写一次"，访存 uops 大幅下降
 *   - MR/NR 可用 -DMR=.. -DNR=.. 覆盖做扫描实验（默认 6×16，
 *     12 个累加向量 + 2 个 B 向量，恰好填满 16 个 ymm 寄存器）
 *   - 行/列边界用补零的 padding 行（A 多算若干零行）与 padding 列
 *     （B 补零，D 写入 padding 列但永不读出）处理，使循环边界全是
 *     编译期常量，编译器才能完全展开并做寄存器分配
 *   - 循环序 jt(列块) 外层 × i0(行块) 内层：每个 B 微面板（K×NR≈128KB）
 *     在 L2 中被所有行块复用，整份打包 B 只需从 L3 读一次
 *
 * 读入/写出与 v2 相同（各进程私有 mmap + 并行解析 + Allgatherv 拼装 B；
 * 各进程格式化自己的行块后 pwrite 到各自字节区间）。
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

#ifndef MR /* 微核行数 */
#define MR 6
#endif
#ifndef NR /* 微核列数（int 数，16 = 2 个 ymm） */
#define NR 16
#endif
#define NPAD (((N + NR - 1) / NR) * NR) /* D/Cl 的行宽：补零到 NR 的倍数 */

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

static void *xcalloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz);
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

/* 打包 B：布局 Bp[jt*K + k*NR + n]，jt 为 NR 列的块起点（jt 即列号，是 NR 倍数）。
 * 尾块右侧补零，使所有微面板定宽 NR。 */
static void pack_b(const int *restrict B, int *restrict Bp)
{
    for (int jt = 0; jt < NPAD; jt += NR) {
        int *restrict p = Bp + (size_t)jt * K;
        for (int k = 0; k < K; k++) {
            const int *restrict src = B + (size_t)k * N + jt;
            for (int n = 0; n < NR; n++)
                p[n] = (jt + n < N) ? src[n] : 0;
            p += NR;
        }
    }
}

/* GCC 向量扩展：8×int32 = 一个 ymm；memcpy 会被编译为单条 vmovdqu */
typedef int v8si __attribute__((vector_size(32), may_alias));

static inline v8si vload(const int *p)
{
    v8si r;
    memcpy(&r, p, sizeof r);
    return r;
}

static inline void vstore(int *p, v8si v)
{
    memcpy(p, &v, sizeof v);
}

/* 寄存器微核：D 块（MR 行 × NR 列，行宽 NPAD）+= A 的 MR 行 × 打包 B 面板。
 * MR×(NR/8) 个 v8si 累加器全部展开驻留 ymm 寄存器，跨越整个 K 循环——
 * 纯标量写法会被 GCC 溢出到栈（实测 np=1 反而变慢 60%），必须显式向量化。
 * A 行块、D 行块的 padding 行/列为零或弃写。 */
static void micro(const int *restrict a, const int *restrict bp, int *restrict d)
{
    v8si acc[MR][NR / 8];
    for (int i = 0; i < MR; i++)
        for (int v = 0; v < NR / 8; v++)
            acc[i][v] = vload(d + (size_t)i * NPAD + v * 8);
    for (int k = 0; k < K; k++) {
        const int *restrict b = bp + (size_t)k * NR;
        v8si bv[NR / 8];
        for (int v = 0; v < NR / 8; v++)
            bv[v] = vload(b + v * 8);
        for (int i = 0; i < MR; i++) {
            const int av = a[(size_t)i * K + k];
            const v8si avv = {av, av, av, av, av, av, av, av}; /* vpbroadcastd */
            for (int v = 0; v < NR / 8; v++)
                acc[i][v] += avv * bv[v];
        }
    }
    for (int i = 0; i < MR; i++)
        for (int v = 0; v < NR / 8; v++)
            vstore(d + (size_t)i * NPAD + v * 8, acc[i][v]);
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

    int rows_pad = ((my_rows + MR - 1) / MR) * MR; /* 行数补到 MR 的倍数 */

    int *B = xmalloc((size_t)K * N * sizeof *B);
    int *Bp = xmalloc((size_t)NPAD * K * sizeof *Bp); /* 打包后的 B */
    int *Al = xcalloc((size_t)rows_pad * K, sizeof *Al); /* 尾部零行=padding */
    int *Cl = xmalloc((size_t)my_rows * NPAD * sizeof *Cl);
    int *Dl = xcalloc((size_t)rows_pad * NPAD, sizeof *Dl);
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
               (size_t)my_rows * K); /* Al 行宽=K，行间无 padding，可一次解析 */
    parse_ints(buf + line_off[LINE_B0 + my_first_b], buf + fsize,
               B + dsp_bn[rank], (size_t)my_rows_b * N);
    for (int i = 0; i < my_rows; i++) /* Cl 行宽=NPAD，逐行解析 */
        parse_ints(buf + line_off[LINE_C0 + my_first + i], buf + fsize,
                   Cl + (size_t)i * NPAD, N);
    double t_parse = MPI_Wtime() - p0;
    double t_read = t_scan + t_parse;

    /* ---- 1d. 就地 Allgatherv：各进程贡献已在 B + dsp_bn[rank] 处，拼成完整 B ---- */
    double c1 = MPI_Wtime();
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_INT, B, cnt_bn, dsp_bn, MPI_INT,
                   MPI_COMM_WORLD);
    double t_comm = t_table + (MPI_Wtime() - c1);

    /* ---- 2. kernel：先 D 初值=C，再 pack B，最后 jt×i0 双层微核 ---- */
    double k0 = MPI_Wtime();
    for (int i = 0; i < my_rows; i++)
        memcpy(Dl + (size_t)i * NPAD, Cl + (size_t)i * NPAD,
               (size_t)N * sizeof *Dl);
    double pk0 = MPI_Wtime();
    pack_b(B, Bp);
    double t_pack = MPI_Wtime() - pk0;
    for (int jt = 0; jt < NPAD; jt += NR) /* B 面板（K×NR≈128KB）驻 L2 被所有行块复用 */
        for (int i0 = 0; i0 < rows_pad; i0 += MR)
            micro(Al + (size_t)i0 * K, Bp + (size_t)jt * K,
                  Dl + (size_t)i0 * NPAD + jt);
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
        const int *row = Dl + (size_t)i * NPAD;
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
    double tmax[6] = {t_read, t_comm, t_kernel, t_pack, t_write, t_total};
    double tred[6];
    MPI_Reduce(tmax, tred, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("np=%d  read=%.3f  comm=%.3f  kernel(max)=%.3f(pack=%.3f)  write=%.3f  TOTAL(max)=%.3f s\n",
               nproc, tred[0], tred[1], tred[2], tred[3], tred[4], tred[5]);
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
    free(Bp);
    free(Al);
    free(Cl);
    free(Dl);
    free(cnt_bn);
    free(dsp_bn);

    MPI_Finalize();
    return 0;
}
