# MPI_matrix：可实验的加速方法与工程实施清单

调研日期：2026-10-25。代码审阅基线：[`445a8b696d336352bdee03192bdef1a1319dfd72`](https://github.com/HSrockstar/MPI_matrix/tree/445a8b696d336352bdee03192bdef1a1319dfd72)。

本文对照当前代码、既有优化记录、原始论文、作者技术博客及库的官方文档，整理 **55 个实验方向**。其中包含新数值算法、数据布局与通信算法、现有实现的进一步优化，以及仅在条件变化后值得做的研究候选。55 个方向不等于 55 个全新的矩阵乘法算法，也不代表它们都能提速。

**建议先修正正确性和测量问题，再依次实验：精确 int8 VNNI、首次累加免读 D、解析直接打包、SIMD 结果压缩、发送前打包 B、整数格式化、分段 FP32，以及 split-K。** Strassen、替代基和通信规避算法保留为后续研究方向。

本文只提交调研文档，不改变计算程序。没有在目标 i9-13900K 上完成新方案的端到端性能测试。下文明确区分源码事实、独立数值检查、理论推导和待验证的性能假设。

阅读顺序：先看第 2 节的前置修正，再按第 10 节推进实验；第 3–8 节是方法目录，第 9 节可直接用于实现验收，第 13 节提供原始资料链接。

## 1. 项目边界与当前实现

### 1.1 约束决定了优化目标

依据仓库的[作业要求](../作业要求.md)：

| 项目 | 约束及其影响 |
| --- | --- |
| 运算 | `D = A × B + C`，M=1023、K=2047、N=1023 |
| 正确性 | 整数结果精确一致；近似量化、容差比较不适用 |
| 输入输出 | 文本输入和文本输出都计入成本，不能用事先生成的二进制文件替代正式输入 |
| 并行 | MPI 消息传递；不能用 OpenMP、pthreads 或进程共享矩阵缓冲区加速 |
| 主要规模 | 1/2/4 个进程，重点比较 2 核和 4 核 |
| 目标机器 | 仓库记录为 i9-13900K、128 GB、Ubuntu 22.04、Open MPI 4.1.2；实际实验仍需记录当时配置 |
| 编译 | 作业说明仍写 GCC 10.5，当前 Makefile 指定 GCC 12、`-O3 -march=native -mavxvnni`，两处信息需要统一 |

旧日志将输入值域描述为 `[-99,99]`；作业要求本身仅写“整数”，没有给出这个界限。因此所有低位宽方案必须在解析时验证值域，或者明确其只支持经过验证的作业数据。不能把日志中的特定数据性质当作所有输入的契约。

由维度可直接算得：输入共有 **5,234,691 个整数**，输出有 **1,046,529 个整数**，矩阵乘法有 **2,142,244,863 次标量乘加**。若把一次乘和一次加分别计为一次操作，可报告约 4.284 GOP，不能把整数吞吐率直接写成实测浮点 FLOPS。

仓库日志记录输入文件 17,806,675 字节、标准输出文件 7,266,389 字节；这些是该附件的记录值，不是格式允许的最大长度。本次未重新下载整个输入附件核验值域及字节数。

### 1.2 已经实现的工作，避免重复推荐

当前 [`mpi_matmul.c`](https://github.com/HSrockstar/MPI_matrix/blob/445a8b696d336352bdee03192bdef1a1319dfd72/mpi_matmul.c) 已包含：各 rank 私有 mmap、分片换行扫描、换行位置 Allgatherv、分布式 B 解析、非阻塞 Allgatherv、int16 成对打包、AVX-VNNI `vpdpwssd`、6×16 微核、手写汇编、行超级块、向量 B 打包、匿名大页提示、四位数字表和并行 pwrite。

默认 `KS=2048`，实际 K=2047，所以只有 **一个条带**。代码虽然保留双缓冲结构，默认运行没有多条带通信与计算重叠。`ceil(2047/2)=1024=KS/2`，因此默认会走完整的手写汇编微核；不能因为 K 小于 KS 就判断它只走尾核。

已有日志报告最后版本约 T1=39–42 ms、T2≈23 ms、T4=15–18 ms。这些是仓库历史结果，本次没有复现。现有测量不足以证明已经达到平台极限。

旧实验中的 PGO/LTO、预取、展开、非临时存储、两遍输出格式化等负结果应保留，但只约束当时的数据类型、布局和代码版本。切换 int8 后，工作集和指令瓶颈会变，部分结论可以有条件复测。

## 2. 在任何速度比较前修正的问题

下列位置均以审阅基线为准；函数名比行号更适合作为后续定位依据。

| 编号 | 源码证据 | 影响及处理 |
| --- | --- | --- |
| B01 | `xalloc_huge`，约 98–118 行，小于 2 MiB 或 mmap 失败时回退 malloc | 注释声称等价 calloc，实际不保证清零。Dl 的第一次微核会加载旧内容，Al 的补 K 和补行也依赖零。必须显式初始化被读取的元素，或改成首次覆盖写微核，并单独填好输入 padding |
| B02 | `pack_a_pairs`，约 466 行，将 int 转换结果左移 16 位 | 高半字为负数的位模式可能使 signed 左移溢出，属于 C 未定义行为。用 `uint32_t` 拼接，再通过明确的位模式复制传给向量；不要靠编译器恰好生成预期指令 |
| B03 | `issue_stripe` 和主循环，约 625、727–742 行 | 两个未完成的 Iallgatherv 共用并改写 scnt/sdsp。MPI 要求相关参数数组在请求完成前保持有效且不被修改。每个在途请求独立保存 counts/displacements。默认单条带不触发，减小 KS 后会暴露 [S18] |
| B04 | 溢出守卫与 int32 回退，约 751–795 行 | A/B 已先解析到 int16；截断后再转 int32 无法恢复原值。int32 累加也不能解决真实的 int32 结果溢出。需要窄化前范围校验，以及 int64 参考/回退或明确拒绝超界输入 |
| B05 | `pack_b16_range` 的向量绝对值归约 | `abs(-32768)` 的 int16 位模式不能当正的 signed int16 最大值使用。当前 ±99 不触发；通用 int16 路径必须处理这一边界 |
| B06 | 默认汇编由 `ASM16 && KS==2048` 选择，约 508 行 | 内含固定 6 行、16 列、步长及循环长度；没有完整的 MR/NR/ISA 守卫。修改宏不等于正确生成新微核。Makefile 的全局 ISA 标志也不构成运行时自动回退 |
| B07 | `parse_fast_*`，约 153–238 行；调用约 706 行 | 快路径针对短整数，调用却以整个文件末尾为 end。应验证每行恰好 K/N 个数、标记、值域和 token 结束位置；三位数、长 token、CRLF、缺换行等不能默默走不一致的路径 |
| B08 | `put_int` 的 `g_est[32]`，约 283–305 行 | 对完整 int32 值域，位长 L 可能为 32，现表索引越界。±20,062,746 的专用结果界内不触发；若声称通用 int32 输出，需要修表并测试 INT_MIN |
| B09 | 输出约 846 行，一次 pwrite | 应处理 EINTR 和短写，使用循环写完；偏移和长度计算用足够宽的类型。输入输出不能是同一文件或同一 inode，避免截断输入映射 |
| B10 | 计时约 855–863 行及 `bench_ab.sh` | `%.3f` 秒只有 1 ms 精度，对 15 ms 任务约为 6.7% 的量化步长。提高精度并保存原始样本；kernel 桶含 B 解析/等待/打包，不能再与 read、comm、pack 相加当总耗时 |

**B01 的独立复核：** 在本次分析环境中提取原分配器，以 `MALLOC_PERTURB_=165` 运行，4 进程配置对应的 Dl 分配首字为 `0x5a5a5a5a`。这证明小块路径不保证清零，未执行完整 MPI 程序。默认 NPAD=1024 时，np=4 的 rows_pad=258，Dl=1,056,768 字节；np=2 的 rows_pad=516，Dl=2,113,536 字节，刚过 2 MiB。按进程数改变分配路径会掩盖问题。

## 3. 数值算法：精确性、适用条件和最小实验

优先级：**高**＝当前目标优先实现；**中**＝前面收益落地后比较；**低**＝研究性原型；**条件**＝值域、规模或硬件改变后再做。所有速度方向均为假设。

### A01｜uint8 × int8 → int32 的 AVX-VNNI【高】

当前 `vpdpwssd` 每个 int32 lane 合并两个 int16 乘积。改成 VEX 编码的 `vpdpbusd`，每 lane 合并四个字节乘积；不能误用需要 AVX-512 的 EVEX 形式，也不能假设 AVX-VNNI 等于支持 `s8×s8` 的 AVX-VNNI-INT8。[S01][S02][S03]

令 `U=A+128`，保持 B 为有符号 int8，按列预计算 `sB[j]=Σk B[k,j]`，则

```text
D[i,j] = Σk U[i,k] * B[k,j] - 128*sB[j] + C[i,j]
```

这是精确整数恒等式，没有量化误差。对已经验证的 ±99 输入：

```text
|AB+C| <= 2047*99*99 + 99 = 20,062,746
|Σ U*B| <= 2047*227*99   = 46,002,231
|128*sB| <= 128*2047*99  = 25,939,584
```

这些界均远小于 INT32_MAX。补 K 到 2048 时 B 填 0，U 的逻辑零填 128；补 N 列及列补偿均为 0。补偿只做一次；若按 stripe/rank 分段，可在每段使用该段列和，不能每段重复扣全局列和。

最小实现：先保留 6×16 输出 tile，把 A 改为四字节广播、B 改为每列四个相邻 K 的字节排列；列和融合到 B 解析/打包。再独立比较补偿放在首次 accumulator 初始化、最后 epilogue、输出格式化三处的差异。

理论上点积指令数量约减半，A/B 存储及通信载荷也可约减半；**这不是整程序 2 倍加速保证**。一个 K=2048、Ntile=16 的 B 面板由 64 KiB 降至 32 KiB，可能改善缓存容量压力，但仍要给 A、D 和其他数据留下空间。

验收：先 bit-exact 检查，再测纯核、parse+pack、通信及完整时间；检查反汇编确实使用 VEX `vpdpbusd`，并确认没有意外 int16 饱和中间步骤。

### A02｜AVX2 绝对值／符号变换 int8【中，兼容回退候选】

对 ±99 值域使用 `abs(A)` 作为 unsigned 字节，`sign(B,A)` 作为 signed 字节，有 `abs(a)*sign(b,a)=a*b`。采用 `vpabsb → vpsignb → vpmaddubsw → vpmaddwd(ones) → vpaddd`。

`vpmaddubsw` 每对乘积的最大绝对值为 `2*99*99=19602<32767`，所以本方案不发生 int16 饱和。直接把 A 偏移 128 后接旧 AVX2 maddubs 则没有这个保证。值域包含 -128 时要重新处理符号取反，不能直接泛化。[S01][S03]

最小实验：做同布局 AVX2 微核，与现有 int16 maddwd 和 A01 比较。代价是每行额外的符号操作，可能输给 VNNI；它的意义在于给不支持 VNNI 的机器提供精确低位宽候选。

### A03｜高低位拆分：两个窄位宽点积【低】

取 `U=A+128=lo+16*hi`，lo、hi 均在 0…15，分别计算与 signed B 的点积，最后整型合并并补偿。可用于研究 `vpmaddubsw` 无饱和路径、4-bit 查表或更窄位宽硬件；两条 GEMM/点积路径及位移相加的成本在现有 VNNI 上大概率不划算。不要把 int4 原生乘法指令假设成 13900K 已支持。

验收：验证拆分恒等式、所有中间界限，以及拆分/合并均计时；这是位宽算法的备选实验，不是 A01 的默认实现。

### A04｜按 K 分段的精确 FP32 GEMM【中】

FP32 能精确表示绝对值不超过 2²⁴ 的整数。若一个点积块满足 `Kb*amax*bmax < 2^24`，每个乘积及任意部分和都在精确范围内，普通 IEEE FP32 乘加可保持整数精确性。对 ±99，保守 `Kb<=1711`；优先试 K 分成 1024 和 1023 两段。

每段从零累加，**分别转换为 int32 后再用整数相加并加 C**。不要把两个块继续以 FP32 相加。1024 段绝对界为 10,036,224；完整 K 的界超过 2²⁴。数学证明由本项目值域直接推导；有限域库利用受界浮点运算的实践可参考 [S14]，FLINT 当前开发文档也描述了类似分段思想 [S15]。

最小实现：单线程 SGEMM 基线或手写 AVX2 FMA 核；转换、分段、打包和启动全部纳入总时间。关闭 TF32/BF16 等降精度模式，不能使用会扩大中间值的快速双线性算法后仍沿用本界。

本次独立检查中，全为 99、K=2047 的点积，整数参考是 20,062,647，逐次完整 FP32 累加得到 20,062,312，分段法得到精确答案。这个反例说明“随机样本看起来正确”不足以批准完整 SGEMM 路径。

### A05｜精确 FP64 GEMM【低，质量与性能参照】

当前值域下所有普通点积中间整数都远小于 2⁵³，可使用顺序 BLAS DGEMM 并精确转回整数。它提供独立且成熟的比较对象；转换成本和较低的每寄存器并行度可能不利。完整范围证明仍必须包括 C 与库内部计算模式。[S14]

### A06｜一层 Strassen，在线性组合时融合打包【中偏低】

先只试一层：8 个子乘法变 7 个，理想乘法量减少 12.5%，同时增加矩阵加减与中间结果流量。按实际长方形维度拆分，不要把整个问题扩成 2048×2048 方阵。

参考 BLIS 上的 Strassen/ABC 做法，把输入和差融入 pack，把输出组合融入微核，避免显式保存所有大临时矩阵。[S06] np=4 时本地 M 只有约 256，再递归容易让子 GEMM 太小。先做本地一层，再评估跨 rank 调度；7 个大任务分给 4 个 rank 会不均衡，需细分输出 tile。

普通一层和差可能达到 ±198，A01 的 s8 输入不再直接适用。必须选择 int16 叶子、重新拆分位宽，或单独证明其他路径。比较基线应是优化后的 A01，不能只与旧 int32 三重循环比较。

### A07｜Strassen–Winograd 变体及低临时空间调度【低】

用 7 次子乘法、15 次标准块加减的方案与 A06 对照，进一步比较重用临时区、就地调度、不同子乘法顺序。[S08][S09] 加法数量减少不保证内存流量同步减少；更长的和差链会扩大值域，必须逐表达式证明。先在小矩阵上以 int64 验证，再替换叶子微核。

### A08｜替代基变换 + 公共子表达式消除 + 存储调度【低】

在更稀疏的基下执行双线性算法，再转换回原基；将基变换与 pack/epilogue 融合，并按依赖图复用中间空间。2023 年的工程论文和 2026 年 SPAA 自动生成框架提供了可追踪的路线。[S10][S11]

最小实验：选一个 2×2、秩 7 的方案，比较“普通 Strassen / 替代基 / 替代基+CSE”三种完整成本。转换回标准输出、工作区初始化、整数位宽扩张全部计入。论文在其他 GEMM 平台上的比例不能移植成这里的预期收益。

### A09｜长方形双线性算法、AlphaTensor 与更小乘法秩【低】

用 `[U,V,W]` 表示算法，筛选更匹配本地长方形规模的基例，如 3×3 秩 23、其他长方形分块，生成融合 pack 的代码。[S07][S12] AlphaTensor 有普通算术与有限域算法；不能把它全部排除为 GF(2)，也不能直接把模 2 的方案用于本题。

2026 年的 4×4 秩 48 方法也值得列入候选，但论文要求底层环存在 2 的逆元。[S13] 对整数直接截断除 2 不合法：须通过统一缩放并精确恢复、有理数中间量，或证明合适的二进制浮点精确界。额外位宽和转换可能抵消节省的乘法。

最小实验先做**算法系数证书**：对所有输入基矩阵对，以精确整数/有理数展开验证输出张量，再测小规模代码生成结果。只看“乘法次数更少”不进入性能结论。

### A10｜Winograd/Waksman 内积恒等式【低】

将相邻 K 配对，预计算 `alpha[i]=Σt A[i,2t]*A[i,2t+1]`、`beta[j]=Σt B[2t,j]*B[2t+1,j]`，则

```text
(AB)[i,j] = Σt (A[i,2t]+B[2t+1,j])*(A[i,2t+1]+B[2t,j])
            - alpha[i] - beta[j] + 奇数 K 的最后一项
```

它减少标量乘法，增加大量加法，而且把 A/B 混在一起，削弱常规 GEMM 的重用和字节点积能力；±198 的和也不再适合 signed int8。最小实验仅做 int16 小核与普通核比较，不优先重写 MPI 框架。历史原始资料见 [S16]，工程实现背景见 [S15]。

### A11｜位平面 popcount 与 Four Russians 查表【条件】

可把 signed int8 分为 8 个位平面，以 AND+popcount 得到位乘积计数，再按位权组合；符号位权是 -128。完整 8×8 位平面组合需要多达 64 类积，转置、位打包和加权恢复也有成本。[S17]

Four Russians 可对短 K 组的 B 行子集和建表，A 的位模式选表；在整数上必须用整数和，不能直接使用 GF(2) 的 XOR。表大小随组长指数增长。[S19] 最小实验取 t=4/8、少量 tile，包含建表成本。低比特或二值数据有价值；当前 ±99 稠密数据不优先。

### A12｜模数分解、CRT 重构与受界浮点有限域 GEMM【条件】

用多个模数计算结果，再用中国剩余定理恢复有符号整数；模数乘积需严格大于 `2*max|D|`，每个模数下仍需证明乘加和延迟约简安全。[S14][S15] 这是精确大整数线性代数的成熟方向；当前结果只需约 25 位，原生 int32 已足够，转换、约简、重构很可能成为净开销。

最小实验只在值域/矩阵维度扩大后开展，比较“直接 int64 / 分块 FP64 / 多模数低位宽”。Ozaki-II 等研究属于相近的分解重构思路，不能把多精度任务的加速倍率当作当前 int8 任务的收益。[S45]

### A13｜结构检测、稀疏或低位宽主体 + 精确残差【条件】

若以后输入确有大量零块、重复行列、精确低秩或少数超界异常值，可在解析阶段检测并选择跳零、去重、结构化乘法，或 `B=Bsmall+Bresidual` 后做稠密主体加稀疏残差。借鉴 FBGEMM 的融合与异常值处理思路时必须保留精确残差，不能引入量化舍入。[S39]

当前日志描述的普通随机稠密矩阵不支持这些结构假设。检测、索引构建、分支及重组必须计时；没有稀疏度或结构证据时停止该方向。不能利用固定答案或针对输入哈希返回预存结果。

### A14｜Kronecker 编码、大整数乘法、FFT/NTT【条件，当前不建议实现】

将多个系数嵌入大整数或多项式，借助大整数/卷积算法再提取结果，是精确代数中的方法族；需要隔离系数、处理有符号数和进位、避免非目标交叉项污染。[S15] 它不意味着普通稠密矩阵能直接做几次 FFT 就完成精确乘法。

如果将来元素本身有数百至数千位，可用成熟大整数库作端到端原型。对当前小整数，编码膨胀、变换和提取开销缺乏胜过原生 VNNI 的依据。APA/边界秩算法同样需额外精确恢复证明，列入第 11 节的暂缓项。

## 4. 本地计算、布局和缓存：10 个实验方向

本节的基本方法来自 Goto/BLIS 的分层组织、GEMMFIP 的融合思路和可复现的 CPU 微核工程实践。[S04][S05][S20] 表中具体应用方式是针对本项目提出的实验设计。

| ID / 优先级 | 改动与最小实验 | 收益观察点、成本与失败条件 |
| --- | --- | --- |
| **K01 高：beta=0 首次覆盖写** | 首个 K 块将 accumulator 置零，或初始化为补偿/C；后续块才加载 D 累加。独立比较显式 memset 与首核覆盖写 | 同时消除 B01 对 Dl 初值的依赖和首轮读流量；不能误删输入 padding 初始化，不能每块重复加 C |
| **K02 高：解析直接生成 A 打包格式** | 每解析一行立即 pack，进一步直接输出 pair/quartet，不保留完整 Al。当前代码是解析全部行后再打包全部行，与“趁行在 L1”注释不一致。另可比较 GEMMFIP 式“首次计算时同步 pack，后续复用” | 观察 parse+pack+compute 合计与峰值工作集；失去通用回退输入时须能重新解析或保留宽位原值；首次计算的 pack 排列必须与复用阶段一致 |
| **K03 中：A 按微核行组交错** | 将 A 从 `[row][Kgroup]` 改为 `[rowtile][Kgroup][MR]`，同一 K 的 6 行广播源相邻；对照原 4096 字节行步长 | 减少独立访存流和地址计算；增加 pack 成本。比较总耗时，不能只看热缓存微核 |
| **K04 中：破除 2 的幂行步长冲突** | 比较 4096 与 4160 字节等保持 64B 对齐的物理步长；逻辑 N/K 不变。A、D 分别测试 | 检查 L1/L2 miss、DTLB 与 cycles；硬编码汇编必须同步修改。Intel 官方也建议某些情形给 leading dimension 增加一个缓存行 [S30] |
| **K05 中：重新选择 MR×NR** | int8 后试 4×16、6×16、8×8、4×24 等小集合，生成正确专用核；先估算 16 个 YMM 寄存器预算 | accumulator 约需 `MR*ceil(NR/8)` 个寄存器，还要留给 A/B/临时量。淘汰出现 spill 的方案；不同形状是调参，不重复计作新算法 |
| **K06 中：联合 K 分块与行超级块** | 在正确的多 stripe 实现上比较 Kb=256/512/1024/2048、行组约 48/96/192；先粗筛再细调 | Kb 小有利工作集和重叠，却多次读写 D、增加 MPI 启动与补偿；以总时间选优。旧 int16 下 KS 负结果不能代替新 int8 实验 |
| **K07 中：软件流水与小幅展开** | 交错下一组 B load/A broadcast 与当前点积，比较展开 1/2/4；用反汇编和 llvm-mca 诊断依赖与端口压力 [S31] | 热循环 cycles 应下降且代码膨胀有限；预加载不得越界。llvm-mca 是静态模型，不能代替实机测量 |
| **K08 中：按行超级块流式生产结果** | 单条带完整 B 就绪后，解析/打包少量 A、读取对应 C、完成全部 K×N、立即格式化这一行组，复用小 D 缓冲 | 降低 Al/Apair/Cl/Dl 常驻量，让计算结果趁热消费。必须完成全 K 后才输出；格式化表也会挤占缓存。与多 stripe 流水存在调度取舍 |
| **K09 低：递归缓存组织 / Morton-hybrid** | 在较大维度递归切最大维，叶子仍调用已优化微核；对比仅递归访问顺序与真正转换布局两种成本 | 可适应多级缓存，代价是递归、索引和格式转换；固定小规模可能不如手工块。相关精确计算研究见 [S32] |
| **K10 中：成熟库与生成核基线** | oneDNN 顺序运行时、FBGEMM、LIBXSMM、顺序 BLAS 分别建立最小适配器；只选择目标 ISA 真正支持的精确路径 | 检查是否发生饱和、浮点 epilogue/requantization、隐藏线程、重复 pack、JIT。一次性任务需计入首次初始化与 JIT；可用离线生成避免在线生成成本，但不能缓存实际输入的计算结果 [S01][S37][S38][S39] |

K10 不是“链接库必然更快”。例如 LIBXSMM 主要提供小矩阵核，大矩阵仍需调用者分块；版本和数据类型不同，实际后端可能不同。oneDNN 应构建 `ONEDNN_CPU_RUNTIME=SEQ`，仍需检查所用 primitive 的确切数据通路；将输出指定为 s32 不自动证明所有内部步骤都保持精确。

## 5. 文本解析与读取：6 个实验方向

当前瓶颈候选是“SIMD 算出候选值后，落栈再逐个压缩”，以及重复扫描与多 rank 缺页。不能根据 simdjson 的 JSON 吞吐率推导本题短整数解析的速度上限。

| ID / 优先级 | 改动与最小实验 | 工程边界与判断标准 |
| --- | --- | --- |
| **P01 高：SIMD 压缩有效结果** | 每 8 个 int16 候选使用 8-bit mask 查 256 项 shuffle 表，将有效值连续写出；每项 16B，表约 4 KiB，再处理另一半 | 消除 `tmp[16]`、逐 bit ctz 和标量写回。先穷举 256 个 mask、0/全选/跨半区，保证顺序与输出计数；AVX2 pshufb 在 128-bit lane 内工作 [S21][S22] |
| **P02 中：更宽分类和十进制合并** | 以 32/64 字节块分类数字、符号和分隔符，利用 shuffle 与 maddubs 的 10/1 权重解码一两位数 | 数字乘 10 的小范围不会触发 maddubs 饱和；真正困难是跨块 token 与负号继承。输入更宽不保证有效值产出更快 [S21][S23] |
| **P03 中：短整数 SWAR 专用路径** | 对已验证的 1/2 位及可选负号使用标量宽加载、掩码和固定算术，减少每数分支；与 P01 对照 | 小 token 下构建 SIMD mask 的固定成本可能更大。用真实位数/负号分布，包含跨缓存行和尾部；不能复用日志中已失败的同一“跳空格 SIMD”实现冒充新方案 |
| **P04 低：合并换行扫描与整数解析** | 各 rank 按字节片解码，修复边界 token，以 prefix count 得到全局序号，再将 A/C 分发给行 owner，B 分发或复制 | 减少文本完整预扫描，但引入 token 边界状态、标记识别和 Alltoallv。先用小文件证明无丢失/重复；通信成本超过扫描节省即停止 |
| **P05 中：只预取本 rank 需要的映射页** | 对比需求缺页、当前整文件 POPULATE_READ、选择性预填扫描片及 A/C/B 区间；按页对齐并检查返回值 | 每个 rank 整文件预填会重复建页表；本地 page cache 热不等于页表已热。不要把预填移出计时伪装加速。API 语义见 [S29] |
| **P06 低：pread 到对齐私有缓冲区** | 对照 mmap：按已知区间或受控块读取，解码同一数据；分别测热缓存与明确记录的冷启动环境 | 多一次拷贝可能换来更低缺页/映射成本；完整输入只有约 18 MB，结论取决于系统。不能在共享机器上为跑分随意清空全局页缓存 |

P01 与 K02 可以组合：先得到连续有效数，再直接构造微核输入。第一版优先保持清晰的解码/pack 边界，确认收益后再融合，避免一次同时引入解析与排列错误。

## 6. MPI 分解与通信：13 个实验方向

当前按输出行分配、复制 B 的 1D 方案，在 2/4 rank 下可能仍然最优。复杂算法的意义是提供可测的替代方案，不能仅凭渐近通信量判定胜负。

| ID / 优先级 | 可实施方案 | 收益来源与主要成本 |
| --- | --- | --- |
| **M01 高：发送前 pack B** | 每 rank 解析自己的 B K 段时生成最终 VNNI 布局，并同时算列和；Allgatherv 收到 rank-major 的 packed 段，本地直接消费 | 将“每个 rank 重复 pack 完整 B”改成全局只 pack 一份。rank 边界按 K-group 对齐；跨段 accumulator 需正确保留，避免每段重置 |
| **M02 中：补齐 K 分片后用等长 Allgather** | K 补到 2048；np=4 每 rank 512 个 K，np=2 每 rank 1024 个 K，最后补零。和 M01 组合 | 简化 counts/displacements，可能触发更好的 collective 路径；等长本身不保证快。np=3 等不整除情况保留 Allgatherv 或独立规则 |
| **M03 中：每 rank 独立解析完整 B** | 去掉 B collective，所有 rank 从只读输入映射解析并 pack 自己的 B 副本 | 用重复 CPU 工作换通信与同步。2/4 rank、热页缓存时值得测；B 解析成本上升，不能假设必优 |
| **M04 中：单 rank 解析/pack B 后广播** | root 一次生成 packed B、列和和元数据，其他 rank 可先处理 A/C，再 Bcast/Ibcast | 减少 pack 重复和调度复杂度，但集中解析可能成为瓶颈；与 M01、M03 三方比较，不预设赢家 |
| **M05 中：B 分段环传** | rank 先计算自有 B K 段，Isend/Irecv 到邻居，同时消费收到的下一段；A 保留对应 K 片 | 不必每 rank 长期保存全部 B，可尝试通信重叠；多轮消息、D 更新和尾部延迟可能更贵。先用双缓冲、明确 request 生命周期 [S24] |
| **M06 中：split-K + Reduce_scatter** | 每 rank 拥有一段 K，计算 `T_r=A[:,Kr]B[Kr,:]`，对 T 做 MPI_SUM 并按行分发 D；C 在归约后由输出 owner 加一次 | K 缩短、局部 B 更小；代价是分发 A 的列片及约 4 MiB int32 部分结果归约。文本按行存储，A 解析/重分布成本不可省略 |
| **M07 低：按输出列分配** | rank 拥有 B 的列块和完整 A，独立计算 D 列块 | 与行划分对称的数学结构，文本输出却按行，可能需要转发或大量碎片写。必须把结果重新组织成本计入 [S24] |
| **M08 低：2D SUMMA / Cannon / Fox** | np=4 构造 2×2 网格，以面板广播或循环移位提供 A/B，输出 owner 负责本地 C | 可能降低复制量，增加 communicator、边界与文本重分布。优先简单 SUMMA；Cannon 处理初始错位及矩形补齐，Fox 处理行广播/列轮转；HSUMMA 层次化仅在更大拓扑下有意义 [S24][S25] |
| **M09 条件：2.5D/3D 复制换通信** | 使用额外副本，在不同层计算 K 子集并归约部分 D；先给出内存与通信模型再实现 | 面向更多 rank/多节点更有价值；np=4 的层数选择太少，额外归约未必抵得上节省的通信 [S26] |
| **M10 条件：CARMA / COSMA 的形状自适应分解** | 借鉴递归划分最大维、依据内存预算选择复制与归约的调度，使用现有精确叶子核 | 比固定 2D 网格更适合不同矩形形状；原库的浮点、GPU、线程和通信实现不能直接假设符合本作业。先做两侧 MPI 消息传递原型和数据量模型 [S27][S28] |
| **M11 中：显式推动非阻塞通信** | 先修 B03；在计算行组边界调用 MPI_Test，比较每行组或约 50/100/200 μs 检查一次 | 非阻塞不等于后台必然推进。以等待时间下降和总时间下降共同验收；调用过密会拖慢微核，额外 progress 线程不纳入本作业 [S33] |
| **M12 中：选择 collective 的实际实现** | 查询本机 `ompi_info` 和 collective 组件，比较适用的 ring、recursive doubling 等策略及消息分段 | `coll_tuned_*` 未必控制 Iallgatherv，后者可能由 libnbc 提供。按安装版本查询选项，不能照抄 Open MPI 5/6 的算法编号到 4.1 [S34] |
| **M13 低：按端到端成本平衡行块** | 保持连续输出行，依据输入字节数、pack 量、计算行数建立权重，或实验静态细粒度块循环划分 | 当前算术工作近似均匀，1023 的余数仅一行；首先应修不合理绑核。仅当 rank 时间明显失衡时启用；块循环会增加输出偏移管理 |

### 6.1 M01 的布局协议应先写清楚

一个便于验证的协议是：`[rank_segment][Npanel][Kgroup][column_in_panel][4 bytes]`，元数据记录 `global_k_begin`、真实 K 长度、补齐长度和列和。各 rank 段可直接拼接，但它**不等于**把原 `[Npanel][global_Kgroup]` 布局原样 Allgatherv。先让微核逐 rank 段累加，确认总成本，再决定是否需要派生 datatype 或接收重排。

派生 datatype 不保证零拷贝，可能在 MPI 内部再次打包。调试时用 index-coded B，逐字节验证每个 `(k,j)` 落到正确 lane；性能阶段只使用最少必要元数据。int8 后实际数据约 2 MiB，任何额外完整重排都值得单独计时。

### 6.2 M06 必须正确处理归约

所有 rank 按相同扁平顺序保存部分 D；`recvcounts[r]` 等于 rank r 的真实行数乘实际通信行宽，发送缓冲区长度等于这些计数的总和。若用 NPAD 通信，需要所有 rank 同意同一 padding；也可只归约 N 列。MPI_SUM 的每个局部和、任意归约子树部分和都要满足范围界。当前 ±99 可用 int32，更大值域用 int64。

如果已经把 A 偏移 128，每个 K 分片先减去自己的列补偿后再归约，或归约所有列和后统一扣一次。C 只能添加一次，不能每个 rank 的部分结果里都带 C。

## 7. 整数格式化与写出：7 个实验方向

| ID / 优先级 | 改动与最小实验 | 成本、正确性及验收 |
| --- | --- | --- |
| **O01 高：完整 jeaiii 算法对照** | 引入独立、小型整数转十进制实现，和当前 `%10000 + 四位表` 比较；使用真实 D 值分布 | jeaiii 的定点乘法生成数字与当前“类似 jeaiii 风格”不是同一实现。包含负号、零和位数边界，遵守源码许可证 [S35][S36] |
| **O02 中：小数字表与静态常量表** | 比较 200B 两位表与 40KB 四位表；分别比较运行时构表和编译期只读常量 | 四位表减少迭代却占据较多 L1。构表目前每 rank 都做，静态表会增加二进制/取页成本；完整一次性运行才能决定 |
| **O03 中：AVX2 批量数字长度和转换** | 一次处理 4/8 个整数，整数阈值判断长度、倒数乘法求十进制块、shuffle 产生 ASCII | 不用浮点 log10 判断边界。2026 年 IFMA 论文可提供思路，但 AVX-512 IFMA 核不能直接在 13900K 使用 [S40] |
| **O04 中：小批次长度前缀和 + 融合写出** | 对 4/8 个值局部算长度、prefix offsets，再以安全的宽存储写入，直接追加分隔符 | 避免全矩阵“两遍长度+再转换”的额外大数组。宽写可以重叠但须有充分尾部余量，不能越到其他 rank 的输出区域；与 K08 组合单独测试 |
| **O05 低：Exscan 求输出起始偏移** | 用每 rank 已格式化字节数做 MPI_Exscan，替代当前所有 rank 收集全部长度 | 小 p 下未必快；rank 0 的 Exscan 接收值未定义，必须显式设 0。`#D\n` 只归 rank 0 的字节块计入一次 [S41] |
| **O06 低：MPI-IO 与 POSIX pwrite 对照** | 比较 MPI_File_write_at、MPI_File_write_at_all 与现有 pwrite；按文件系统区分结果 | 集体 I/O 的聚合器在并行文件系统有用，在单机小文件上可能增加开销。包含 open/close、datatype 构造及同步成本；零输出 rank 仍参与所需 collective [S42][S43] |
| **O07 低：输出批大小与定长预分配** | 对照“一 rank 一大块”与按行组缓冲写；知道总长度后测试 ftruncate/必要预分配 | 多次小写有系统调用成本，一大块需较大缓冲。既有 fallocate 负结果保留；只在流式方案改变写入模式后重测。短写/EINTR 重试是正确性要求，不能作为可选速度开关 |

当前 pwrite 完成表示数据已交给内核，未做 fsync，不代表持久介质已落盘。正式比较保持同一语义；若另测持久化时间，所有候选都纳入相同 flush 流程。输出需保持 `#D`、整数、空格和换行的规定格式。

## 8. 分配、硬件和构建：5 个工程实验方向

| ID / 优先级 | 实验 | 观察点与限制 |
| --- | --- | --- |
| **E01 高：统一对齐与访问契约** | 对微核缓冲使用至少 64B 对齐的分配器；区分“对齐”与“清零”，比较 loadu 和有证据的 aligned load | malloc 只保证语言要求的对齐，不能当作任意 32/64B 对齐。非临时存储通常需更严格对齐，且 D 很快会被读取，不应默认启用 |
| **E02 中：按缓冲角色评估 THP** | 当前大缓冲已 madvise；进一步比较按 2 MiB 向上取整、小缓冲合并、普通页，记录实际 AnonHugePages | MADV_HUGEPAGE 是提示，成功返回不保证全部采用大页；过度取整增加物理内存和缺页成本。只改本进程分配策略，不改全局内核设置 [S44] |
| **E03 中：私有 arena 与缓冲复用** | 预分配少量对齐大区，按生命周期复用临时区；K08 可降低峰值，再重新评估 THP | 避免大量 mmap/malloc 与重复清零；不同逻辑数组不可因复用发生读写冲突，MPI 未完成的区域不能回收。保持每 rank 私有 |
| **E04 高：实际物理核绑定** | 用 lscpu/hwloc 查 P/E 核及 SMT 拓扑，1 rank/物理核，np=1/2/4 用可比较的 P 核集合，记录 report-bindings | CPU 编号不能写死，容器 CPU 配额和允许 CPU 集合也需记录。不得杀其他用户任务或修改全局频率/电源策略来制造结果 [S46] |
| **E05 中：编译器、生成代码与反馈优化** | 固定源码哈希，比较 GCC/Clang 可用版本及相同 ISA；PGO 仅在新热点形成后重测，确认 profile 被实际读入 | 旧 PGO 名称错位导致“静默未生效”，有效 PGO 的 np=4 结果反而较差。LTO 对单文件收益有限；运行时 ISA 分派需安全 baseline 翻译单元，不能只加一个 if 判断 [S47] |

## 9. 实现时的 60 项工程检查清单

以下复选框表示**未来实施时需要验收的事项**，不是已经完成的代码修复。每个实验只勾选与其相关的条目，并保留验证证据。

### 9.1 数据契约与解析

- [ ] **Q01** 在窄化前验证整数范围；确认 ±99 是所用输入的事实，超界时拒绝或进入经过验证的宽位路径。
- [ ] **Q02** 每行使用自己的结束边界，核对元素数、矩阵标记和总行数，不允许跨行补足缺失元素。
- [ ] **Q03** 明确支持的空格、负号、CRLF、末尾换行规则；不支持的形式应报错，不能静默误读。
- [ ] **Q04** 向量加载不得读出 mmap 有效可访问区；尾部用标量路径或复制到有 padding 的私有临时块。
- [ ] **Q05** 穷举压缩表 mask；验证 128-bit lane 边界、负号跨块、连续分隔符、零及 token 跨 16/32/64 字节边界。
- [ ] **Q06** 若跳过旧 Al/Bs 中间数组，保留可定位的原始行偏移和明确的回退策略，避免校验失败后只能使用已截断数据。

### 9.2 整数范围与数值语义

- [ ] **Q07** 使用 int64 或更宽精确参考实现；守卫表达式先升宽再乘，避免守卫本身溢出。
- [ ] **Q08** 分别证明原始点积、偏移点积、补偿、C、Strassen 中间和差的界限，不只验证最终答案范围。
- [ ] **Q09** 测试 signed/unsigned 约定和 `vpdpbusd` 操作数顺序；禁止混用饱和与非饱和指令后沿用同一证明。
- [ ] **Q10** 检查 -128、-32768、INT_MIN 的取绝对值和取负；范围不支持这些值时在入口明确拒绝。
- [ ] **Q11** FP32 每段独立从零计算、先转整数再归并；记录库的精度模式，确保无隐藏降精度或浮点补偿舍入。
- [ ] **Q12** 快速双线性算法的除法、基变换和系数在目标数域上成立；用精确系数证书验证，随机测试只能辅助。

### 9.3 布局、补齐与边界

- [ ] **Q13** 分开记录逻辑 M/N/K、物理 lda/ldb/ldc、row padding、K-group 长度及实际分配字节数。
- [ ] **Q14** 第一次会读取的 D 元素均已初始化，或首次核确实覆盖写；不得依赖 malloc 返回零页。
- [ ] **Q15** 补齐 K、N、M 的每一个输入元素都写入正确逻辑零；U=A+128 的补零表示为 128。
- [ ] **Q16** 用小型 index-coded 矩阵验证 pack/unpack 的每个 lane、跨 rank 段、奇数 K 与尾列。
- [ ] **Q17** 改 MR/NR/stride 后重建所有相关核及常量，增加 compile-time 条件或 static assertions，杜绝静默错配。
- [ ] **Q18** 将对齐、容量、初始化分别作为分配器契约；测试强制 malloc 回退和被污染的内存。

### 9.4 C、汇编与 ISA

- [ ] **Q19** 位模式拼接使用 unsigned 运算；避免 signed 左移溢出、不正确的窄化和违反严格别名的指针转换。
- [ ] **Q20** 对齐 load/store 前证明地址对齐；宽输出 store 也必须位于实际已分配缓冲区内。
- [ ] **Q21** 检查内联汇编输入输出、clobber、memory 约束，独立汇编检查 callee-saved 寄存器、栈对齐和 ABI。
- [ ] **Q22** 反汇编核对 VEX/EVEX、字节有符号性、spill、重复 load、循环次数和尾核实际调用路径。
- [ ] **Q23** 若提供通用二进制，启动和 dispatcher 用安全 ISA 编译；仅在检测支持后调用目标核，外部库也需匹配。
- [ ] **Q24** 调试构建运行 ASan/UBSan 及尾部 guard 测试；手写汇编的访问不一定被 sanitizer 覆盖，补独立边界测试。

### 9.5 MPI 生命周期与协议

- [ ] **Q25** 每个在途请求拥有独立且稳定的 send/recv 缓冲区及 counts/displacements，完成后才修改或复用。
- [ ] **Q26** 所有 rank 在同一 communicator 上以一致次序调用 collective；错误路径也不能让其他 rank 无限等待。
- [ ] **Q27** `MPI_IN_PLACE` 的本地数据处于自己的正确接收位置；若改普通 send buffer，检查允许的别名关系。
- [ ] **Q28** counts 的单位是元素还是字节写清楚；datatype 的 extent、真实数据尺寸及 displacement 单位分别验证。
- [ ] **Q29** 消息采用匹配的固定宽度 MPI datatype；大文件用 64-bit 偏移，并验证当前 MPI int count 能否容纳。
- [ ] **Q30** 测试非整除分配、最后 stripe、np=3、空片及零计数路径；不支持进程数应及早一致退出。

### 9.6 重叠、归约与跨方案组合

- [ ] **Q31** 用时间线证明 Iallgatherv 与计算实际重叠；记录 MPI_Test 次数及等待时间，不能以非阻塞 API 名称代替证据。
- [ ] **Q32** split-K 的 MPI_SUM 对任意归约顺序都不会溢出；C 与补偿各只计入一次。
- [ ] **Q33** 2D/递归分解核对全局到局部索引、communicator rank 重排、文本输入 owner 和输出 owner。
- [ ] **Q34** 明确 K08 行流式与多 stripe 流水的冲突：没有收齐该行全部 K 贡献时不得格式化最终结果。
- [ ] **Q35** 检查库是否内部创建线程、执行 JIT 或再次打包；要求 MPI-only 时使用顺序运行时并在实验中记录。
- [ ] **Q36** 对 collective 调参确认实际组件和版本；记录策略选择日志，保证选项真的作用于调用的函数。

### 9.7 输出和文件系统

- [ ] **Q37** 验证 0、负数、10 的幂附近、最大允许值及通用路径的 INT_MIN，检查长度表全部索引。
- [ ] **Q38** 输出缓冲按合法最大长度计算，使用 size_t 并检查乘法溢出；保留宽写尾部余量。
- [ ] **Q39** `#D\n` 恰好一次，每行数字个数和分隔符准确；禁止多个 rank 的输出字节范围重叠。
- [ ] **Q40** root 截断文件的顺序与其他 rank 打开/写入正确；输入和输出路径及 inode 不得相同。
- [ ] **Q41** pwrite 循环处理 EINTR、短写及零进展；检查 open、write、close 错误，失败样本不能记为有效速度。
- [ ] **Q42** 明确页缓存完成与 fsync 持久化口径；MPI-IO 集体操作包含所有参与 rank，完成前不释放缓冲。

### 9.8 系统与内存

- [ ] **Q43** 保存 CPU 型号、P/E 核、SMT、允许 CPU 集合、cgroup 配额及每个 rank 的实际绑定。
- [ ] **Q44** 比较 1/2/4 rank 时使用一致类型物理核；排除 rank 被绑到同一物理核的两个逻辑线程。
- [ ] **Q45** 将分配、首次触页、初始化及 madvise 计入正式范围；读取 smaps 等重诊断放在独立分析运行。
- [ ] **Q46** 记录实际大页使用与缺页数，检查 mmap/madvise 失败回退；不能因提示成功就认定 THP 生效。
- [ ] **Q47** arena 的活跃区间不重叠；调试构建能够完整释放或报告各类分配，避免重用后出现悬挂指针。
- [ ] **Q48** 不通过修改全局内核参数、清空他人缓存、终止其他用户任务或额外占用核数获取不公平结果。

### 9.9 测量与统计

- [ ] **Q49** 计时打印至少 6 位小数秒，原始记录不经过提前四舍五入；同时记录内部时间和外部进程 wall time。
- [ ] **Q50** 把 t_cput、parse、pack、MPI wait、format、write 独立标注；重叠阶段和各 rank 最大值不能直接相加。
- [ ] **Q51** A/B 相同预热策略、随机 AB/BA 顺序，按相邻样本成对比较；报告全部有效样本而非最小值。
- [ ] **Q52** shell 启用失败检测，保留 stderr、退出码及验证结果；输出文件每轮唯一，防止读取前次成功结果。
- [ ] **Q53** 记录中位数、分位区间、配对改善及不确定性；微小变化未超过噪声时标为“无结论”。
- [ ] **Q54** 重复测量保持相同输入及输出语义，区分热缓存、首次启动和外部启动成本；不把昂贵步骤移到计时外。

### 9.10 构建、复现与发布

- [ ] **Q55** 每次从已知源码构建，记录 commit、编译器、MPI 版本、flags、宏、二进制 hash，避免使用仓库旧可执行文件。
- [ ] **Q56** PGO 检查 gcda 路径及缺失 profile 警告，训练集覆盖相关进程数；不能把未生效的 PGO 结果当优化证据。
- [ ] **Q57** 对基线和候选都执行老师的 verify_；另以独立参考验证对抗样本，不能只互相比对两个优化实现。
- [ ] **Q58** 每项提交保留最小可回退变更和实验 ID，组合优化前分别保存独立 A/B 结果。
- [ ] **Q59** 记录外部代码许可证、依赖版本和构建方式，确保老师的重新编译环境可复现，缺少 ISA 时行为明确。
- [ ] **Q60** 最终 1/2/4 rank 用同一算法与计时口径重测 S(p)、E(p)，同时报告绝对时间，避免低效单核基线放大加速比。

## 10. 实验执行顺序与通过标准

### 10.1 建立三个互补的测量层次

1. **纯核诊断**：预先准备正确 packed 输入，测 cycles/输出元素、指令数、缓存与 TLB 行为；帮助决定 MR/NR/展开，不用于宣称整程序收益。
2. **内部全流程**：从任务数据准备前开始，到输出完成为止，逐 rank 记录阶段和总时间。当前程序从 MPI_Init 后开始，到汇总 Reduce 前结束，漏掉初始化、末尾汇总、清理与 Finalize，因此不等于严格完整进程时间。
3. **外部完整运行**：由父进程单调时钟测启动 `mpirun` 到退出，单独报告 launcher/MPI 启动的影响。作业要求“整个程序”时，这层结果需要保留；如有老师给定测试器，以该口径为准。

计时点增加本身也有开销，诊断版本与最终版本分开。不要为了把所有 rank 计时对齐而在每个阶段添加 Barrier：这会改变可重叠的执行。报告本地阶段分布以及关键路径，总体指标使用各 rank 总时间的最大值。

### 10.2 最小统计协议

- 基线必须先包含 B01–B10 中与路径相关的修正。不能把修复未定义行为前后的差异直接解释成算法速度差。
- 每个候选先进行一次完整正确性检查及尾部/极值测试，再做至少 30 对相邻 A/B 运行作为初筛。初筛发现差异或波动较大时，再增加样本；没有新疑点不无限重复。
- A/B 使用相同预热次数，随机选择每对 AB 或 BA；保存运行顺序、原始高精度时间、退出码、验证状态和环境。
- 报告两种程序中位数、IQR，以及配对时间比 `T_candidate/T_baseline` 的中位数和 bootstrap 95% 区间。预先固定异常样本规则，系统故障可剔除，普通慢样本不能因为“不好看”删除。
- 建议实用门槛：配对时间比区间整体低于 1，且效果超过已测得噪声。若仅有很小均值变化而区间跨 1，保留“未证明收益”；不宣称提速。
- 改善率写成 `1 - T_candidate/T_baseline`；速度倍数写成 `T_baseline/T_candidate`，避免混用。不同核心数分别报告。
- 真正的全流程瓶颈占比确定后再估收益上限：若某段占比 f、该段提速 s，则理想总加速比为 `1/((1-f)+f/s)`。例如 f=0.5、s=2，仅有约 1.33 倍整体加速；这是示例，不是本项目预测。

建议原始结果表字段：

```text
experiment_id, source_commit, binary_sha256, compiler, mpi_version,
flags, np, rank_bindings, MR, NR, KS, SB, input_sha256,
pair_id, order, cache_mode, internal_total_s, external_wall_s,
parse_s, pack_s, compute_s, wait_s, format_s, write_s,
exit_code, verify_ok, notes
```

性能计数器只在单独诊断中采集，并确认权限与事件支持。多 rank 的 `perf stat mpirun ...` 不一定按预想覆盖每个计算进程；应核实继承/远端启动方式，必要时每 rank 分别采集再聚合。计数器运行与普通计时不要直接混为一组样本。

### 10.3 建议分四轮推进

| 轮次 | 实验范围 | 完成条件 |
| --- | --- | --- |
| R0：可信基线 | B01–B10、Q01–Q60 中相关基础项、提高计时精度、真实物理核绑定 | np=1/2/4 正确，内存污染测试通过；小规模支持 np=3/边界或明确拒绝；留存高精度基线 |
| R1：高收益候选 | A01 + K01；随后 K02、P01、O01 各自独立 A/B | 每项先独立验收再组合，得到新的稳定基线；不要一次改核、解析和通信后无法归因 |
| R2：减少数据移动 | M01/M02，与 M03/M04 对照；K03/K04/K06/K08、P05、O02/O04 | 用实际阶段时间决定优先级；确认组合收益没有被新打包、缺页或写出开销抵消 |
| R3：替代算法 | A04、M06、M05、K10，再考虑 A06–A10、M08–M10 | 至少证明某个实际目标配置总时间更好才保留；低优先级原型先证明精确性与粗粒度性能，不直接重写主程序 |

组合时重点检查：A01 改变 K 分组和补零；A06–A09 扩大输入范围，不能无条件调用 A01；M01 改变 B 段布局；M06 改变 C/补偿归属；K08 改变生命周期；P01 的 LUT 和 O02 的数字表会竞争缓存。多个单独有效的优化不保证叠加有效。

## 11. 已考虑但不建议作为当前主线的方法

| 方法 | 本次判断与重新考虑条件 |
| --- | --- |
| GPU/cuBLAS/CUTLASS、AMX、AVX-512 IFMA/VBMI | 当前目标硬件或作业配置不具备对应条件。换机器后可重新评估，但传输、初始化及文本 I/O 都要计入 |
| 普通完整 SGEMM、TF32/BF16/FP16 近似量化 | 不能保证当前任务逐元素精确。A04 是有界分段方案，不能与这些近似路径混同 |
| 直接 GF(2) 的 M4RI 或 AlphaTensor 模 2 方案 | 运算域不同；需要位平面整数恢复或选择普通算术方案，见 A09/A11 |
| APA、border-rank、Bini 类方法 | 2026 年研究包含从修改后的 APA 得到精确乘积的理论，但普通 APA 只给近似结果。未为本输入完成精确恢复及中间位宽证明，不批准直接替换；未来可作为 A09/A14 的独立研究分支 [S48] |
| 更低渐近指数、Coppersmith–Winograd/激光方法 | 渐近算术界不能推出本规模机器时间；本次没有找到可直接与 256×2047×1023 本地 VNNI 核竞争的已验证实现 |
| 稀疏 GEMM、低秩近似、随机采样、近似矩阵乘法 | 输入结构未获证实；低秩近似和采样不满足精确输出。仅保留 A13 的可验证精确结构分支 |
| MPI+OpenMP、MPI 共享窗口、后台计算/进展线程 | 与当前禁止共享内存优化及固定核心数的任务边界不符；不作为默认方案 |
| MPI-4 持久 collective | 当前 Open MPI 4.1.2 所对应接口能力不能假设支持 MPI-4；且单次大乘法缺少摊销空间。重复同形状任务并升级运行时后再评估 |
| 缓存输入的二进制版本、预存 packed B、预存答案 | 若正式计时绕过本次文本输入、pack 或实际计算，会改变任务；只可作为明确标注的微核诊断，不能进入成绩比较 |
| O_DIRECT、io_uring、大量异步小写 | 未发现当前一次小文件输出需要这些机制的证据；先测 I/O 等待。若将来有真实长 I/O，仍须计入 completion，保证对齐及线程约束 |
| 无条件软件预取、non-temporal store、巨幅循环展开 | 现有日志已有无益或失败结果；仅当 int8/新布局后的计数器显示相关瓶颈时再做有针对性的复测 |
| 全矩阵格式化“两遍法” | 旧日志已有负收益；保留 O04 的小批次局部前缀和，避免重复生成整幅中间输出 |
| 修改全局 THP/频率、清空全局缓存、额外抢核 | 改变共享运行环境或比较资源，不纳入算法优化方案 |

## 12. 本次验证记录与调研停止条件

### 12.1 已完成的独立检查

检查环境为本次分析容器显示的 Xeon Platinum 8573C、GCC 13.3，支持 AVX-VNNI；不是仓库目标 13900K。未安装/运行完整目标 MPI 环境，未运行老师的 verify_，也没有新方案的端到端加速数据。

| 检查 | 覆盖与结果 | 能证明什么 |
| --- | --- | --- |
| A01 VEX int8 点积 | K∈{1,2,3,4,7,511,512,513,1023,2047}，每种 100 组、8 lanes，共 8000 个精确输出；随机 ±99 与极值；与 int64 参考一致 | 验证本次 quartet 排列、偏移补偿和奇数 K 补齐的小原型；反汇编核对 VEX vpdpbusd。未验证整套 MPI/pack 集成 |
| A02 AVX2 符号变换 | 同样 10 种 K、每种 100 组，共 1000 个点积，与 int64 一致 | 验证 ±99 内无饱和路径；不代表完整 int8 值域都安全 |
| A04 分段 FP32 | 1000 个点积按 1024 分段后正确；完整 FP32 的全 99、K=2047 反例已复现 | 支持数值界和算法步骤；未测试具体 BLAS 后端内部行为 |
| A10 配对内积恒等式 | 4 组不同矩阵形状，含 K=2047，共 70 个整数输出，与直接乘法一致 | 检查公式和奇数尾项，没有性能结论 |
| B01 分配初始化 | 原 allocator 的小块 malloc 路径在内存扰动下返回非零数据 | 证明“等价 calloc”的注释不成立；没有声称完整 MPI 程序已复现错误答案 |

数值小检查采用固定伪随机种子 1234567，覆盖全正/全负极值和不整除边界。点积原型仅使用标准 C 与 intrinsics；VNNI 检查使用 GCC 的 `-O3 -march=x86-64-v3 -mavxvnni`，A02/A04 对照使用 `-O3 -mavx2 -fno-tree-vectorize -ffp-contract=off`。这些是正确性检查，不是性能数据。后续正式实现应将对应测试纳入仓库可复现的测试程序，而非仅引用本文结果。

### 12.2 “搜不到新算法”的实际含义

无法证明世界上不存在遗漏或未来出现的新算法。本次采用明确边界：**CPU、MPI-only、1/2/4 rank、给定矩阵规模、精确整数、文本全流程**。先扩展方法族，再沿论文引用、作者代码及工程实现查找可落实的变体；同一方法的块尺寸、线程数或 ISA 宽度变化不重复算独立算法。

扩展阶段新增了精确低位宽的两条实现、分段 FP32、Winograd 内积、替代基、长方形双线性算法、位串行/CRT、大整数编码、split-K、CARMA/COSMA、递归缓存组织及解析/格式化融合。最终补检如下：

| 补检轮次 | 代表检索式与范围 | 去重后的结论 |
| --- | --- | --- |
| 1 | `exact int8 matrix multiplication CPU AVX VNNI vpdpbusd signed compensation algorithms`；`exact matrix multiplication bit serial Four Russians CRT Winograd integer CPU` | 返回 VNNI 实现、位平面、CRT、GF(2) 和新硬件变体，归入 A01–A03、A11–A14，没有新增当前可实施方法族 |
| 2 | `MPI matrix multiplication communication avoiding SUMMA Cannon CARMA COSMA hierarchical`；`libxsmm gemm packing fusion AVX2 integer` | 归入 M05、M08–M12 和 K02/K10；层次化调度与生成核是已有类别的实现变体 |
| 3 | `jeaiii itoa integer to string`；`MADV_POPULATE_READ mmap performance`；`nonblocking collectives arrays counts displacements not modified completion`；`WojciechMula parsing-int-series SIMD` | 进一步核实 O01、P01/P05 和 B03/Q25，未新增独立算法类别 |

至此在上述边界下达到**本轮检索饱和**，应把时间转向实验。若值域、硬件、节点数、重复调用次数或输入格式变化，重新打开第 11 节中对应的条件项。

## 13. 主要资料与对应用途

以下为原始论文、作者维护的代码/博客或官方文档，共 48 个编号条目。某些条目同时列论文和作者实现，以便落地；检索日期均为 2026-10-25。库文档可能比目标机器的安装版本新，API/选项必须以本机版本复核。

- **[S01]** [oneDNN：Nuances of int8 Computations](https://uxlfoundation.github.io/oneDNN/dev_guide_int8_computations.html)。核对 VNNI、AVX2 中间饱和、有符号输入补偿及精度边界。
- **[S02]** [LLVM AVX-VNNI intrinsic 源码](https://clang.llvm.org/doxygen/avxvnniintrin_8h_source.html)。核对 `_mm256_dpbusd_epi32` 的 unsigned/signed 四字节分组语义与 target 属性。
- **[S03]** [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)。核对 maddubs、maddwd、sign、abs 和点积指令语义；以具体 ISA 版本为准。
- **[S04]** Goto、van de Geijn，[Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS.pdf)，ACM TOMS，2008。分层缓存、打包与微核设计。
- **[S05]** Xu、Van Zee、van de Geijn，[GEMMFIP: Unifying GEMM in BLIS](https://arxiv.org/abs/2302.08417)，2023。融合操作与减少打包成本的参考。
- **[S06]** Huang 等，[Implementing Strassen's Algorithm with BLIS](https://arxiv.org/abs/1605.01078)，2016 技术报告。将快速算法的线性组合融入 BLIS 打包和微核。
- **[S07]** Huang、Rice、Matthews、van de Geijn，[Generating Families of Practical Fast Matrix Multiplication Algorithms](https://arxiv.org/abs/1611.01120)，IPDPS 2017。长方形双线性算法、U/V/W 表示和代码生成。
- **[S08]** Winograd，[On multiplication of 2×2 matrices](https://research.ibm.com/publications/on-multiplication-of-2-2-matrices)，1971。Strassen–Winograd 的原始算法背景。
- **[S09]** [Duke：SC 1998 Strassen 工程研究](https://users.cs.duke.edu/~alvy/papers/sc98/index.htm)。缓存、数据布局、递归阈值与调度的工程问题。
- **[S10]** [Pebbling Game and Alternative Basis for High Performance Matrix Multiplication](https://epubs.siam.org/doi/10.1137/22M1502719)，SIAM J. Scientific Computing，2023。替代基与存储调度。
- **[S11]** Bruker、Schwartz、Vaknin，[An Automatic Framework for High Performance Alternative Basis Fast Matrix Multiplication](https://www.cs.huji.ac.il/~odedsc/papers/BrukerSV-SPAA26.pdf)，SPAA 2026。替代基、CSE 与 pebble 调度的自动生成框架。
- **[S12]** Fawzi 等，[Discovering faster matrix multiplication algorithms with reinforcement learning](https://www.nature.com/articles/s41586-022-05172-4)，Nature，2022；[作者代码及系数](https://github.com/google-deepmind/alphatensor)。选择正确运算域，验证算法系数。
- **[S13]** Dumas 等，[Fast matrix multiplication via recursive ⟨4×4×4:48⟩ algorithms into practice](https://arxiv.org/abs/2609.12027)，2026 预印本。48 乘法与替代基；特别注意 2 可逆的数域条件。
- **[S14]** [FFLAS-FFPACK 项目](https://linbox-team.github.io/fflas-ffpack/)；[Dumas、Giorgi、Pernet，ISSAC 2004 论文](https://membres-ljk.imag.fr/Clement.Pernet/Publications/ffpack_issac04.pdf)。精确有限域线性代数借助高性能数值核。
- **[S15]** [FLINT 作者站：整数矩阵乘法](https://fredrikj.net/fdoc/html/algorithms/linear-algebra/integer-matrix-multiplication.html)及[小模数乘法](https://fredrikj.net/fdoc/html/algorithms/linear-algebra/small-modulus-matrix-multiplication.html)，2026 开发文档。用于方法分类和实现入口；页面注明部分文本由 LLM 生成，因此本文不以其单独作为正确性证明，数值界已独立推导，关键方法另列原始论文。
- **[S16]** [Stanford 1970 矩阵算法技术报告](https://i.stanford.edu/pub/cstr/reports/cs/tr/70/157/CS-TR-70-157.pdf)。Winograd 内积法历史参考；本文给出的配对恒等式另行检查。
- **[S17]** [maltanar/gemmbitserial](https://github.com/maltanar/gemmbitserial)。signed/unsigned 位串行整数 GEMM 的工程实现。
- **[S18]** [MPI 3.1：Nonblocking Collective Operations](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node126.htm)。counts、displacements、缓冲区和完成语义；直接对应 B03。
- **[S19]** Albrecht、Bard、Hart，[Efficient Multiplication of Dense Matrices over GF(2)](https://arxiv.org/abs/0811.1714)，后发表于 ACM TOMS；[M4RI 实现](https://github.com/malb/m4ri)。Four Russians 方法和运算域限制。
- **[S20]** Salykova，[Beating OpenBLAS for a Single Matrix Multiplication](https://salykova.github.io/matmul-cpu)，2025，作者技术博客。CPU 微核、寄存器预算、打包与测量实践；其浮点结果不直接等同于本题整数结果。
- **[S21]** Wojciech Muła，[parsing-int-series](https://github.com/WojciechMula/parsing-int-series)。连续十进制整数 SIMD 解析的作者代码。
- **[S22]** [Algorithmica：SIMD Shuffling](https://en.algorithmica.org/hpc/simd/shuffling/)。shuffle/permute 与 AVX2 lane 限制。
- **[S23]** [hunyadi/simdparse](https://github.com/hunyadi/simdparse)。向量十进制解析与位权合并的代码参考。
- **[S24]** David Bindel，[Cornell CS 5220：Parallel matrix multiply](https://www.cs.cornell.edu/courses/cs5220/2020fa/lec/2020-10-22-matmul.html)，2020。行列分解、环传、SUMMA 和 Cannon 的教学推导。
- **[S25]** van de Geijn、Watts，[SUMMA: Scalable Universal Matrix Multiplication Algorithm](https://www.cs.utexas.edu/~rvdg/abstracts/SUMMA.html)，1995 技术报告/1997 论文；[HSUMMA 研究](https://arxiv.org/abs/1306.4161)，2013。二维面板广播与层次化变体。
- **[S26]** Solomonik、Demmel，[Communication-optimal parallel 2.5D matrix multiplication and LU factorization algorithms](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-72.html)，2011。额外内存换较少通信。
- **[S27]** Lipshitz，[Communication-Avoiding Parallel Recursive Algorithms for Matrix Multiplication](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2013/EECS-2013-100.pdf)，2013。CARMA、CAPS 与矩形递归调度。
- **[S28]** Kwasniewski 等，[Red-blue pebbling revisited: near optimal parallel matrix-matrix multiplication](https://arxiv.org/abs/1908.09606)，SC 2019；[COSMA 代码](https://github.com/eth-cscs/COSMA)。内存预算、矩阵形状和通信调度。
- **[S29]** [Linux man-pages：madvise(2)](https://www.man7.org/linux/man-pages/man2/madvise.2.html)。POPULATE_READ、页对齐与错误语义。
- **[S30]** [Intel oneMKL：Coding Techniques](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2026-0/coding-techniques.html)。数据对齐及 leading dimension padding 的实践建议；仅借鉴缓存原则，具体 Linux 实现需实测。
- **[S31]** [LLVM：llvm-mca](https://llvm.org/docs/CommandGuide/llvm-mca.html)。机器码吞吐、依赖和资源压力的静态诊断。
- **[S32]** [Cache-oblivious Matrix Multiplication for Exact Factorisation](https://arxiv.org/abs/1705.04807)，2017。Morton-hybrid 与精确计算中的递归缓存组织。
- **[S33]** [MPI Performance Guidelines：Progress](https://mpi-performance-guidelines.github.io/progress.html)。弱进展模型、MPI_Test 和通信重叠实验。
- **[S34]** [Open MPI：Available Collective Components](https://docs.open-mpi.org/en/main/tuning-apps/collectives/components.html)及[tuned 组件](https://docs.open-mpi.org/en/main/tuning-apps/collectives/tuned.html)。区分 blocking/nonblocking 的实际实现；目标 4.1 需本机查询。
- **[S35]** James Anhalt，[jeaiii/itoa](https://github.com/jeaiii/itoa)。整数转 ASCII 的作者实现。
- **[S36]** Junekey Jeon，[Faster integer formatting — James Anhalt's algorithm](https://jk-jeon.github.io/posts/2022/02/jeaiii-algorithm/)，2022。定点乘法生成十进制数字的算法分析。
- **[S37]** [oneDNN：Build Options](https://uxlfoundation.github.io/oneDNN/dev_guide_build_options.html)。`ONEDNN_CPU_RUNTIME=SEQ` 等构建选项。
- **[S38]** [LIBXSMM：Matrix Multiplication](https://libxsmm.github.io/libxsmm/documentation/libxsmm_mm.html)及[项目文档](https://libxsmm.github.io/libxsmm/documentation/)。小 GEMM、生成核、数据类型和调用者分块责任。
- **[S39]** Khudia 等，[FBGEMM: Enabling High-Performance Low-Precision Deep Learning Inference](https://arxiv.org/abs/2101.05615)，2021；[Meta 工程博客](https://engineering.fb.com/2018/11/07/ml-applications/fbgemm/)，2018。融合、生成核和低位宽工程；机器学习精度容忍不能代替整数精确证明。
- **[S40]** Champagne Gareau、Lemire，[Converting an Integer to a Decimal String in Under Two Nanoseconds](https://doi.org/10.1002/spe.70079)，Software: Practice and Experience，2026；[作者说明](https://lemire.me/blog/2026/05/18/simd-accelerated-integer-to-string-conversion/)。IFMA 十进制并行生成及硬件边界。
- **[S41]** [MPI：Exclusive Scan](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node143.htm)。rank 0 接收缓冲结果未定义，需要显式处理。
- **[S42]** [Open MPI：MPI_File_write_at_all](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_File_write_at_all.3.html)。带显式偏移的集体文件写入。
- **[S43]** [NERSC：I/O Libraries](https://docs.nersc.gov/performance/io/library/)。并行文件系统上的 MPI-IO 工程实践。
- **[S44]** [Linux Kernel：Transparent Hugepage Support](https://docs.kernel.org/admin-guide/mm/transhuge.html)。大页提示、对齐、实际使用与内存放大。
- **[S45]** [Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique](https://journals.sagepub.com/doi/10.1177/10943420261467787)，2026。低位宽子积与 CRT 重构的多精度方向。
- **[S46]** [Open MPI：Tuning FAQ](https://www.open-mpi.org/faq/?category=tuning)。绑核、映射及运行配置。
- **[S47]** [GCC：Function Multiversioning](https://gcc.gnu.org/onlinedocs/gcc/Function-Multiversioning.html)。多版本函数及目标 ISA；实际属性/CPU 检测名称按使用的 GCC 版本确认。
- **[S48]** Dumitriu 等，[Fast and Stable Arbitrary Precision Approximate Matrix Multiplication](https://arxiv.org/abs/2610.13905)，2026 预印本。用于区分普通近似 APA 与经修改后精确恢复的理论路线，未据此宣称当前项目可提速。

仓库内的历史对照资料：[优化日志](../优化日志.md)、[第三轮调研](../调研报告-第三轮.md)、[PPT 素材](../PPT素材-优化历程.md)、[构建文件](../Makefile)、[A/B 脚本](../bench_ab.sh)。实现时应把新实验结果追加到独立记录，避免改写历史结论而丢失适用条件。



[S01]: https://uxlfoundation.github.io/oneDNN/dev_guide_int8_computations.html
[S02]: https://clang.llvm.org/doxygen/avxvnniintrin_8h_source.html
[S03]: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
[S04]: https://www.cs.utexas.edu/~flame/pubs/GotoTOMS.pdf
[S05]: https://arxiv.org/abs/2302.08417
[S06]: https://arxiv.org/abs/1605.01078
[S07]: https://arxiv.org/abs/1611.01120
[S08]: https://research.ibm.com/publications/on-multiplication-of-2-2-matrices
[S09]: https://users.cs.duke.edu/~alvy/papers/sc98/index.htm
[S10]: https://epubs.siam.org/doi/10.1137/22M1502719
[S11]: https://www.cs.huji.ac.il/~odedsc/papers/BrukerSV-SPAA26.pdf
[S12]: https://www.nature.com/articles/s41586-022-05172-4
[S13]: https://arxiv.org/abs/2609.12027
[S14]: https://linbox-team.github.io/fflas-ffpack/
[S15]: https://fredrikj.net/fdoc/html/algorithms/linear-algebra/integer-matrix-multiplication.html
[S16]: https://i.stanford.edu/pub/cstr/reports/cs/tr/70/157/CS-TR-70-157.pdf
[S17]: https://github.com/maltanar/gemmbitserial
[S18]: https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node126.htm
[S19]: https://arxiv.org/abs/0811.1714
[S20]: https://salykova.github.io/matmul-cpu
[S21]: https://github.com/WojciechMula/parsing-int-series
[S22]: https://en.algorithmica.org/hpc/simd/shuffling/
[S23]: https://github.com/hunyadi/simdparse
[S24]: https://www.cs.cornell.edu/courses/cs5220/2020fa/lec/2020-10-22-matmul.html
[S25]: https://www.cs.utexas.edu/~rvdg/abstracts/SUMMA.html
[S26]: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-72.html
[S27]: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2013/EECS-2013-100.pdf
[S28]: https://arxiv.org/abs/1908.09606
[S29]: https://www.man7.org/linux/man-pages/man2/madvise.2.html
[S30]: https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-windows/2026-0/coding-techniques.html
[S31]: https://llvm.org/docs/CommandGuide/llvm-mca.html
[S32]: https://arxiv.org/abs/1705.04807
[S33]: https://mpi-performance-guidelines.github.io/progress.html
[S34]: https://docs.open-mpi.org/en/main/tuning-apps/collectives/components.html
[S35]: https://github.com/jeaiii/itoa
[S36]: https://jk-jeon.github.io/posts/2022/02/jeaiii-algorithm/
[S37]: https://uxlfoundation.github.io/oneDNN/dev_guide_build_options.html
[S38]: https://libxsmm.github.io/libxsmm/documentation/libxsmm_mm.html
[S39]: https://arxiv.org/abs/2101.05615
[S40]: https://doi.org/10.1002/spe.70079
[S41]: https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node143.htm
[S42]: https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_File_write_at_all.3.html
[S43]: https://docs.nersc.gov/performance/io/library/
[S44]: https://docs.kernel.org/admin-guide/mm/transhuge.html
[S45]: https://journals.sagepub.com/doi/10.1177/10943420261467787
[S46]: https://www.open-mpi.org/faq/?category=tuning
[S47]: https://gcc.gnu.org/onlinedocs/gcc/Function-Multiversioning.html
[S48]: https://arxiv.org/abs/2610.13905
