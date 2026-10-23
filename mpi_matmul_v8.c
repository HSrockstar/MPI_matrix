/*
 * mpi_matmul.c — 基于 MPI 的 D = A*B + C（纯消息传递，无共享内存并行）
 * 版本 6（当前最优）：v4（I/O 并行+寄存器微核）之上叠加两项——
 *   a) B 条带流水线（v5）：解析切片 + 非阻塞 MPI_Iallgatherv 与上一条带
 *      计算重叠，通信与 B 解析被计算掩盖（np≥2 时收益）；
 *   b) int16/vpmaddwd 通路（v6）：A 存 int16（行宽 K+1 补零凑对），B 面板
 *      按 k 两两交错打包，一条 vpmaddwd 完成 2k×8j 共 16 次乘加（内部
 *      32 位精度），uops/MAC 约为 int32 微核一半；带溢出守卫，数据超界
 *      自动回退 int32 通路（本作业数据 ±99 不会触发）。
 *   c) 推荐以 make pgo 构建最终二进制（PGO 实测再快 3~5%）。
 *
 * 矩阵规模（固定于作业输入）：
 *   A: M×K = 1023×2047   B: K×N = 2047×1023   C/D: M×N = 1023×1023
 *
 * v5 kernel 数据通路（每条带一轮）：
 *   issue(s+1): 解析条带 s+1 的本进程切片 → Bs[(s+1)%2]（就地）→
 *               MPI_Iallgatherv 非阻塞拼整条带（与后续计算重叠）
 *   wait(s) → pack 条带 s（Bs → Bp 打包面板，仅本条带的 k 行）→
 *   compute(s)：jt(列块)×i0(行块) 扫描，寄存器微核累加 k∈[ks,ke) 到 D
 *   双条带缓冲 + 双请求，避免未完成的收集被覆盖。
 *
 * 读入/写出的并行结构与 v2 起一致（私有 mmap + 并行扫描/解析 +
 * Allgatherv 行表；pwrite 各写各的区间）。
 *
 * 构建: mpicc -O3 -Wall -march=native -o mpi_matmul mpi_matmul.c
 * 运行: mpirun -np <p> ./mpi_matmul <RandMatrix.txt> <result.txt>
 */
#include <fcntl.h>
#include <immintrin.h>
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
#define NPAIR ((K + 1) / 2) /* k 方向两两配对的组数（vpmaddwd 一次算 2 个 k） */

#ifndef KS /* B 条带宽度（k 方向），需为偶数保证打包对齐；实测 128~1024 差异在噪声内 */
#define KS 512
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

/* 大缓冲区专用：2MB 对齐 + MADV_HUGEPAGE 的匿名映射（内容为零，等价 calloc）。
 * 本机实测 4K 首触 672ns/页 → THP 后 328ns/页（减半），且 DTLB 压力大减；
 * 截去首尾使 [al, al+sz) 恰好 2MB 对齐、可整段大页缺页。进程退出即归还，
 * 不走 free()。小于 2MB 的分配退回普通 malloc（THP 无从生效）。 */
static void *xalloc_huge(size_t sz)
{
    const size_t HP = (size_t)2 << 20;
    if (sz < HP)
        return xmalloc(sz);
    const size_t tot = sz + 2 * HP; /* 头尾各留一个 2MB 余量用于对齐 */
    unsigned char *base = mmap(NULL, tot, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED)
        return xmalloc(sz);
    unsigned char *al =
        (unsigned char *)(((uintptr_t)base + HP - 1) & ~(uintptr_t)(HP - 1));
    if (al > base)
        munmap(base, (size_t)(al - base));
    const size_t kept = (sz + 4095) & ~(size_t)4095;
    if (al + kept < base + tot)
        munmap(al + kept, (size_t)((base + tot) - (al + kept)));
#ifdef MADV_HUGEPAGE
    madvise(al, kept, MADV_HUGEPAGE);
#endif
    return al;
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
#ifdef MADV_POPULATE_READ
    /* 一次性预入页表，避免扫描/解析途中散落的惰性缺页（内核 ≥5.14） */
    madvise(p, (size_t)st.st_size, MADV_POPULATE_READ);
#endif
    *out_len = (size_t)st.st_size;
    return (const char *)p;
}

/* simdjson 式并行解析（专用于本作业已验证的格式：单空格/换行分隔，
 * 值域 ±99 即 ≤2 位带符号整数；启动时以 3 连数字断言兜底）。
 *
 * 每 16 字节一块：先算分隔符位图与 token 起始位图，再用三条错位加载
 * (p, p+1, p+2) 在 16 个 32 位通道上同时算出所有 token 的值——
 *   无符号 1~2 位: v = w0 + (w1 是数字 ? 10*w1 : 0)
 *   带 '-':      v = -(w1 + (w2 是数字 ? 10*w2 : 0))
 * 非 token 起始通道为垃圾值，压缩输出只读起始通道（tzcnt 遍历位图）。
 * 跨块 token 判定：块内最高分隔符位置 hi 之后的起始位留到下一块处理。 */
#define DEF_PARSE_FAST(SUFFIX, OUT_T)                                               \
    static const char *parse_fast_##SUFFIX(const char *p, const char *end,          \
                                           OUT_T *out, size_t cnt, int *amax)       \
    {                                                                               \
        size_t c = 0;                                                               \
        int16_t tmp[16] __attribute__((aligned(32)));                               \
        while (c < cnt && (size_t)(end - p) >= 19) {                                \
            const __m128i v0 = _mm_loadu_si128((const __m128i *)p);                 \
            const __m128i v1 = _mm_loadu_si128((const __m128i *)(p + 1));           \
            const __m128i v2 = _mm_loadu_si128((const __m128i *)(p + 2));           \
            const __m128i sp = _mm_or_si128(                                        \
                _mm_cmpeq_epi8(v0, _mm_set1_epi8(' ')),                             \
                _mm_cmpeq_epi8(v0, _mm_set1_epi8('\n')));                           \
            const __m128i dg = _mm_and_si128(                                       \
                _mm_cmpgt_epi8(v0, _mm_set1_epi8('0' - 1)),                         \
                _mm_cmplt_epi8(v0, _mm_set1_epi8('9' + 1)));                        \
            const int m = _mm_movemask_epi8(sp);                                    \
            if (!m) { p += 16; continue; }                                          \
            const int d = _mm_movemask_epi8(dg);                                    \
            if (d & (d >> 1) & (d >> 2))                                            \
                die("3+ digit");                                                    \
            const int s = ~m & ((m << 1) | 1);                                      \
            const int hi = 31 - __builtin_clz((unsigned)m);                         \
            const unsigned done = (unsigned)s & ((1u << hi) - 1);                   \
            const __m256i W0 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v0),           \
                                                _mm256_set1_epi16('0'));            \
            const __m256i W1 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v1),           \
                                                _mm256_set1_epi16('0'));            \
            const __m256i W2 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v2),           \
                                                _mm256_set1_epi16('0'));            \
            const __m256i M1 = _mm256_and_si256(                                    \
                _mm256_cmpgt_epi16(W1, _mm256_set1_epi16(-1)),                      \
                _mm256_cmpgt_epi16(_mm256_set1_epi16(10), W1));                     \
            const __m256i M2 = _mm256_and_si256(                                    \
                _mm256_cmpgt_epi16(W2, _mm256_set1_epi16(-1)),                      \
                _mm256_cmpgt_epi16(_mm256_set1_epi16(10), W2));                     \
            const __m256i Mn = _mm256_cmpeq_epi16(W0, _mm256_set1_epi16(-3));       \
            const __m256i vdir = _mm256_add_epi16(                                  \
                W0, _mm256_and_si256(_mm256_add_epi16(                              \
                                       _mm256_mullo_epi16(W0, _mm256_set1_epi16(9)), \
                                       W1),                                         \
                                     M1));                                          \
            const __m256i vneg = _mm256_add_epi16(                                  \
                W1, _mm256_and_si256(_mm256_add_epi16(                              \
                                       _mm256_mullo_epi16(W1, _mm256_set1_epi16(9)), \
                                       W2),                                         \
                                     M2));                                          \
            const __m256i val = _mm256_blendv_epi8(                                 \
                vdir, _mm256_sub_epi16(_mm256_setzero_si256(), vneg), Mn);          \
            _mm256_store_si256((__m256i *)tmp, val);                                \
            for (unsigned bits = done; bits && c < cnt; bits &= bits - 1) {         \
                const int v = tmp[__builtin_ctz(bits)];                             \
                out[c++] = (OUT_T)v;                                                \
                if (amax) {                                                         \
                    const int a = v < 0 ? -v : v;                                   \
                    if (a > *amax)                                                  \
                        *amax = a;                                                  \
                }                                                                   \
            }                                                                       \
            const unsigned todo = (unsigned)s & ~((1u << hi) - 1);                  \
            p = todo ? p + __builtin_ctz(todo) : p + hi + 1;                        \
        }                                                                           \
        while (c < cnt) {                                                           \
            while (p < end && *p != '-' && (*p < '0' || *p > '9'))                  \
                p++;                                                                \
            if (p >= end)                                                           \
                die("eof");                                                         \
            int neg = 0, v = 0;                                                     \
            if (*p == '-') { neg = 1; p++; }                                        \
            while (p < end && *p >= '0' && *p <= '9') {                             \
                v = v * 10 + (*p - '0');                                            \
                p++;                                                                \
            }                                                                       \
            v = neg ? -v : v;                                                       \
            out[c++] = (OUT_T)v;                                                    \
            if (amax) {                                                             \
                const int a = v < 0 ? -v : v;                                       \
                if (a > *amax)                                                      \
                    *amax = a;                                                      \
            }                                                                       \
        }                                                                           \
        return p;                                                                   \
    }

DEF_PARSE_FAST(i16, int16_t)
DEF_PARSE_FAST(i32, int)

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

/* 四位数字查表（miloyip/branchlut + Lemire fix-digit 风格 itoa）：10000 组
 * "0000".."9999"，整 4 位一组一次 memcpy（编译为单条 32 位 mov），尾组按
 * 总位数折出有效长度后跳过前导零。对 7~8 位的 D 值比两位表再快约一倍。 */
static char g_quad[40000];

static void init_digit_pairs(void)
{
    for (int i = 0; i < 10000; i++) {
        g_quad[4 * i] = (char)('0' + i / 1000);
        g_quad[4 * i + 1] = (char)('0' + (i / 100) % 10);
        g_quad[4 * i + 2] = (char)('0' + (i / 10) % 10);
        g_quad[4 * i + 3] = (char)('0' + i % 10);
    }
}

static size_t put_int(char *dst, int v)
{
    size_t n = 0;
    unsigned u;
    if (v < 0) {
        dst[n++] = '-';
        u = 0U - (unsigned)v;
    } else {
        u = (unsigned)v;
    }
    int d = (u < 10) ? 1 : (u < 100) ? 2 : (u < 1000) ? 3 : (u < 10000) ? 4
          : (u < 100000) ? 5 : (u < 1000000) ? 6 : (u < 10000000) ? 7
          : (u < 100000000U) ? 8 : (u < 1000000000U) ? 9 : 10;
    char *q = dst + n + d; /* 尾后指针，从后往前成组写 */
    while (u >= 10000) {
        const unsigned r = u % 10000;
        u /= 10000;
        q -= 4;
        memcpy(q, &g_quad[4 * r], 4);
    }
    /* 尾组有效位数 = ((d-1) mod 4) + 1，查表项跳过前导零 */
    const int tl = ((d - 1) % 4) + 1;
    memcpy(q - tl, &g_quad[4 * u + (4 - tl)], tl);
    return n + d;
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

/* 打包一条 B 条带：Bs(srows×N, int16) → Bp 面板中 k∈[ks,ks+srows) 的行。
 * 面板布局 Bp[jt*K + k*NR + n]，jt 为 NR 列块起点；尾块右侧补零。 */
static void pack_b_range(const int16_t *restrict Bs, int *restrict Bp, int ks, int srows)
{
    for (int jt = 0; jt < NPAD; jt += NR) {
        int *restrict p = Bp + (size_t)jt * K + (size_t)ks * NR;
        for (int r = 0; r < srows; r++) {
            const int16_t *restrict src = Bs + (size_t)r * N + jt;
            for (int n = 0; n < NR; n++)
                p[n] = (jt + n < N) ? src[n] : 0;
            p += NR;
        }
    }
}

/* 寄存器微核（宏生成定长 k-run 特化：k 循环次数必须是编译期常量，
 * GCC 才会展开——运行期边界的版本实测慢 ~25%）。
 * D 块（MR×NR，行宽 NPAD）+= A 的 MR 行 × Bp 面板 k∈[ks,ks+CNT) */
#define DEF_MICRO(SUFFIX, CNT)                                                       \
    static void micro_##SUFFIX(const int16_t *restrict a, const int *restrict bp,    \
                               int *restrict d, int ks)                              \
    {                                                                                \
        v8si acc[MR][NR / 8];                                                        \
        for (int i = 0; i < MR; i++)                                                 \
            for (int v = 0; v < NR / 8; v++)                                         \
                acc[i][v] = vload(d + (size_t)i * NPAD + v * 8);                     \
        for (int kk = 0; kk < (CNT); kk++) {                                         \
            const int *restrict b = bp + (size_t)(ks + kk) * NR;                     \
            v8si bv[NR / 8];                                                         \
            for (int v = 0; v < NR / 8; v++)                                         \
                bv[v] = vload(b + v * 8);                                            \
            for (int i = 0; i < MR; i++) {                                           \
                const int av = a[(size_t)i * (K + 1) + ks + kk];                     \
                const v8si avv = {av, av, av, av, av, av, av, av};                   \
                for (int v = 0; v < NR / 8; v++)                                     \
                    acc[i][v] += avv * bv[v];                                        \
            }                                                                        \
        }                                                                            \
        for (int i = 0; i < MR; i++)                                                 \
            for (int v = 0; v < NR / 8; v++)                                         \
                vstore(d + (size_t)i * NPAD + v * 8, acc[i][v]);                     \
    }

DEF_MICRO(full, KS) /* 整条带 */
#if K % KS != 0
DEF_MICRO(tail, K % KS) /* 尾条带 */
#endif

/* ---- int16/vpmaddwd 通路（元素 ±99，乘加在 32 位内精确无溢出）----
 * B 面板按 k 两两交错打包：w 的通道为 [b_{k0,j}, b_{k1,j}] 成对，
 * x = 广播 [a_{k0}, a_{k1}]，一条 vpmaddwd 完成 2k×8j 共 16 次乘加，
 * uops/MAC 约为 int32 微核的一半。ks 需为偶数；尾条带行数为奇时
 * 补 k=K 的零行凑对（A 侧 (K+1) 宽、补零已就位）。 */
static void pack_b16_range(const int16_t *restrict Bs, int16_t *restrict Bp16, int ks,
                           int srows, int *bmax)
{
    for (int jt = 0; jt < NPAD; jt += NR) {
        int16_t *restrict base =
            Bp16 + (size_t)(jt / NR) * NPAIR * 2 * 16; /* 面板号 = jt/NR */
        const int p0 = ks / 2, p1 = (ks + srows + 1) / 2;
        for (int p = p0; p < p1; p++) {
            const int r0 = 2 * p - ks, r1 = r0 + 1;
            for (int h = 0; h < 2; h++) {
                int16_t *restrict dst = base + (p * 2 + h) * 16;
                for (int t = 0; t < 8; t++) {
                    const int col = jt + h * 8 + t;
                    int v0 = 0, v1 = 0;
                    if (col < N) {
                        if (r0 < srows)
                            v0 = Bs[(size_t)r0 * N + col];
                        if (r1 < srows)
                            v1 = Bs[(size_t)r1 * N + col];
                    }
                    int a0 = v0 < 0 ? -v0 : v0, a1 = v1 < 0 ? -v1 : v1;
                    if (a0 > *bmax)
                        *bmax = a0;
                    if (a1 > *bmax)
                        *bmax = a1;
                    dst[2 * t] = (int16_t)v0;
                    dst[2 * t + 1] = (int16_t)v1;
                }
            }
        }
    }
}

/* VNNI 点积累加：一条 vpdpwssd 完成 vpmaddwd+vpaddd 两条指令的工作
 * （本机 i9-13900K 的 lscpu 已确认 avx_vnni；无 VNNI 的机器自动回退） */
static inline __m256i dot_acc(__m256i acc, __m256i a, __m256i b)
{
#ifdef __AVXVNNI__
    return _mm256_dpwssd_epi32(acc, a, b);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(a, b));
#endif
}

/* A 预打包为 int32 k-对：Apair[i*NPAIR+p] = {a_{2p}, a_{2p+1}} 两个 int16
 * 打包进一个 int32。微核广播由此改为内存形式 vpbroadcastd [mem]
 * （3/周期、只占 load 口），消除原先"2 次标量加载+合并+寄存器广播
 * （1/周期、独占 port 5）"的瓶颈——这是 VNNI 收益得以释放的前提。
 * 行宽 NPAIR 对：K=2047 奇数，末对高位 = k=K 的补零；尾部零行 = 全零。 */
static void pack_a_pairs(const int16_t *restrict arow, int32_t *restrict aprow)
{
    for (int p = 0; p < NPAIR; p++)
        aprow[p] = (int)(uint16_t)arow[2 * p] | ((int)(uint16_t)arow[2 * p + 1] << 16);
}

#define DEF_MICRO16(SUFFIX, PAIRS)                                                   \
    static void micro16_##SUFFIX(const int32_t *restrict a,                          \
                                 const int16_t *restrict bp, int *restrict d, int p0) \
    {                                                                                \
        __m256i acc[MR][2];                                                        \
        for (int i = 0; i < MR; i++) {                                              \
            acc[i][0] = _mm256_loadu_si256((const __m256i *)(d + (size_t)i * NPAD)); \
            acc[i][1] = _mm256_loadu_si256((                                        \
                const __m256i *)(d + (size_t)i * NPAD + 8));                       \
        }                                                                           \
        for (int pp = 0; pp < (PAIRS); pp++) {                                      \
            const __m256i w0 = _mm256_loadu_si256((                                 \
                const __m256i *)(bp + (size_t)(p0 + pp) * 32));                     \
            const __m256i w1 = _mm256_loadu_si256((                                 \
                const __m256i *)(bp + (size_t)(p0 + pp) * 32 + 16));                \
            for (int i = 0; i < MR; i++) {                                          \
                const __m256i x = _mm256_set1_epi32(                                  \
                    a[(size_t)i * NPAIR + (p0 + pp)]);                               \
                acc[i][0] = dot_acc(acc[i][0], x, w0);                               \
                acc[i][1] = dot_acc(acc[i][1], x, w1);                               \
            }                                                                       \
        }                                                                           \
        for (int i = 0; i < MR; i++) {                                              \
            _mm256_storeu_si256((__m256i *)(d + (size_t)i * NPAD), acc[i][0]);       \
            _mm256_storeu_si256((__m256i *)(d + (size_t)i * NPAD + 8), acc[i][1]);   \
        }                                                                           \
    }

DEF_MICRO16(full, KS / 2)
#if K % KS != 0
DEF_MICRO16(tail, (K % KS + 1) / 2)
#endif

/* 解析条带（ks 起 srows 行）的本进程切片进 Bs（int16，就地），并发起非阻塞拼整 */
static void issue_stripe(const char *buf, size_t fsize, const uint32_t *line_off,
                         int rank, int nproc, int ks, int srows, int16_t *Bs,
                         int *scnt, int *sdsp, MPI_Request *req)
{
    block_layout(nproc, srows, N, scnt, sdsp);
    parse_fast_i16(buf + line_off[LINE_B0 + ks + sdsp[rank] / N], buf + fsize,
                   Bs + sdsp[rank], (size_t)scnt[rank], NULL);
    MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_INT16_T, Bs, scnt, sdsp, MPI_INT16_T,
                    MPI_COMM_WORLD, req);
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

    int rows_pad = ((my_rows + MR - 1) / MR) * MR; /* 行数补到 MR 的倍数 */

    int *Bp = xmalloc((size_t)NPAD * K * sizeof *Bp); /* int32 打包面板（回退用，常态不触页） */
    int16_t *Bp16 = xalloc_huge((size_t)NPAD * NPAIR * 2 * sizeof *Bp16); /* k 交错面板 */
    int16_t *Al = xalloc_huge((size_t)rows_pad * (K + 1) * sizeof *Al); /* 行宽 K+1：末列补零凑 k 对，尾部零行=padding */
    int32_t *Apair = xalloc_huge((size_t)rows_pad * NPAIR * sizeof *Apair); /* A 的 int32 k-对打包（int16 通路用） */
    int amax = 0;
    int *Cl = xalloc_huge((size_t)my_rows * NPAD * sizeof *Cl);
    int *Dl = xalloc_huge((size_t)rows_pad * NPAD * sizeof *Dl);
    int16_t *Bs2 = xalloc_huge((size_t)2 * KS * N * sizeof *Bs2); /* 双条带缓冲(int16) */
    int *scnt = xmalloc((size_t)nproc * sizeof *scnt);
    int *sdsp = xmalloc((size_t)nproc * sizeof *sdsp);

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

    /* ---- 1c. 各进程解析自己的 A/C 行块（B 在流水线中按条带解析） ---- */
    double p0 = MPI_Wtime();
    for (int i = 0; i < my_rows; i++) /* Al 行宽=K+1（末列补零凑 k 对），逐行解析 */
        parse_fast_i16(buf + line_off[LINE_A0 + my_first + i], buf + fsize,
                       Al + (size_t)i * (K + 1), K, &amax);
    /* 趁 Al 行还在 L1，就地合并成 int32 k-对（padding 行全零，直接打包） */
    for (int i = 0; i < rows_pad; i++)
        pack_a_pairs(Al + (size_t)i * (K + 1), Apair + (size_t)i * NPAIR);
    for (int i = 0; i < my_rows; i++) /* Cl 行宽=NPAD，逐行解析 */
        parse_fast_i32(buf + line_off[LINE_C0 + my_first + i], buf + fsize,
                       Cl + (size_t)i * NPAD, N, NULL);
    double t_read = t_scan + (MPI_Wtime() - p0);

    /* ---- 2. B 条带流水线：解析切片+非阻塞收集 与 计算重叠 ---- */
    double k0 = MPI_Wtime();
    /* Dl 累加 A×B（C 的加法融合进最终格式化，省一次 D=C 的 memcpy） */

    const int nstr = (K + KS - 1) / KS;
    MPI_Request req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    double t_bpar = 0.0, t_wait = 0.0, t_pack = 0.0, t_cput = 0.0;

    { /* 发起条带 0 */
        double b0 = MPI_Wtime();
        issue_stripe(buf, fsize, line_off, rank, nproc, 0, KS < K ? KS : K,
                     Bs2, scnt, sdsp, &req[0]);
        t_bpar += MPI_Wtime() - b0;
    }
    for (int s = 0; s < nstr; s++) {
        const int ks = s * KS;
        const int ke = ks + KS < K ? ks + KS : K;
        if (s + 1 < nstr) { /* 解析并预发起条带 s+1（写入另一缓冲，与计算重叠） */
            const int ks2 = (s + 1) * KS;
            const int ke2 = ks2 + KS < K ? ks2 + KS : K;
            double b0 = MPI_Wtime();
            issue_stripe(buf, fsize, line_off, rank, nproc, ks2, ke2 - ks2,
                         Bs2 + ((s + 1) % 2) * (size_t)KS * N, scnt, sdsp,
                         &req[(s + 1) % 2]);
            t_bpar += MPI_Wtime() - b0;
        }
        double w0 = MPI_Wtime();
        MPI_Wait(&req[s % 2], MPI_STATUS_IGNORE);
        t_wait += MPI_Wtime() - w0;

        double c1 = MPI_Wtime();
        int16_t *Bst = Bs2 + (s % 2) * (size_t)KS * N;
        int bmax = 0;
        pack_b16_range(Bst, Bp16, ks, ke - ks, &bmax);
        /* 溢出守卫：|Σ a·b| ≤ amax·bmax·K 加上 |C| 上界，超界回退 int32 通路 */
        const int v16ok =
            ((int64_t)amax * (int64_t)bmax * K + 32767) <= INT32_MAX;
        if (!v16ok)
            pack_b_range(Bst, Bp, ks, ke - ks);
        t_pack += MPI_Wtime() - c1;

        double c2 = MPI_Wtime();
        const int pairs = (ke - ks + 1) / 2;
        for (int jt = 0; jt < NPAD; jt += NR)
            for (int i0 = 0; i0 < rows_pad; i0 += MR) {
                int *d = Dl + (size_t)i0 * NPAD + jt;
                const int32_t *aprow = Apair + (size_t)i0 * NPAIR;
                if (v16ok) {
                    const int16_t *bp =
                        Bp16 + (size_t)(jt / NR) * NPAIR * 2 * 16;
#if K % KS != 0
                    if (pairs == KS / 2)
                        micro16_full(aprow, bp, d, ks / 2);
                    else
                        micro16_tail(aprow, bp, d, ks / 2);
#else
                    micro16_full(aprow, bp, d, ks / 2);
#endif
                } else {
                    const int16_t *arow = Al + (size_t)i0 * (K + 1);
#if K % KS != 0
                    if (ke - ks == KS)
                        micro_full(arow, Bp + (size_t)jt * K, d, ks);
                    else
                        micro_tail(arow, Bp + (size_t)jt * K, d, ks);
#else
                    micro_full(arow, Bp + (size_t)jt * K, d, ks);
#endif
                }
            }
        t_cput += MPI_Wtime() - c2;
    }
    double t_kernel = MPI_Wtime() - k0;
    t_read += t_bpar; /* B 解析计入 read 桶（CPU 口径；与计算重叠部分见 TOTAL） */

    /* ---- 3. 各进程格式化自己的 D 行块（进程 0 的块含 "#D\n" 头） ---- */
    init_digit_pairs();
    double w0 = MPI_Wtime();
    char *obuf = xalloc_huge((size_t)my_rows * N * 12 + 16 + (rank == 0 ? 3 : 0));
    size_t op = 0;
    if (rank == 0) {
        memcpy(obuf, "#D\n", 3);
        op = 3;
    }
    for (int i = 0; i < my_rows; i++) {
        const int *row = Dl + (size_t)i * NPAD;
        const int *crow = Cl + (size_t)i * NPAD; /* 此处补上 +C */
        for (int j = 0; j < N; j++) {
            op += put_int(obuf + op, row[j] + crow[j]);
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
    double tmax[6] = {t_read, t_wait + t_table, t_kernel, t_pack, t_write, t_total};
    double tred[6];
    MPI_Reduce(tmax, tred, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("np=%d  read=%.3f  comm=%.3f  kernel(max)=%.3f(pack=%.3f)  write=%.3f  TOTAL(max)=%.3f s\n",
               nproc, tred[0], tred[1], tred[2], tred[3], tred[4], tred[5]);
    fflush(stdout);

    munmap((void *)buf, fsize);
    /* 大缓冲区（xalloc_huge）为独享 mmap，进程退出即归还，不做 munmap/free；
     * Bp 常态下未被写页（回退通路才用），free 走 glibc 惰性路径即可 */
    free(nl_loc);
    free(nl_all);
    free(line_off);
    free(ncnt);
    free(ndsp);
    free(blk_cnt);
    free(blk_dsp);
    free(Bp);
    free(scnt);
    free(sdsp);

    MPI_Finalize();
    return 0;
}
