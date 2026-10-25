# 55 个实验方向的实现清单

对照 [原调研文档](optimization-research-and-engineering-plan.md)。这里的“覆盖”表示具备可以执行、验证和计时的实验入口，不表示每个方案都更快，也不表示已经复现论文中的全部优化变体。

- **C**：集成到 C/MPI 主程序，已做精确结果验证。
- **模型**：Python 精确算法或 MPI 通信原型，已做小规模验证；不作为优化过的 CPU 核宣称性能。
- **工具**：可执行的环境、构建或测量工具。
- **外部依赖待验收**：有适配器源码，但本次没有对应库，未声称通过编译/运行。

## 数值算法 A01–A14

| ID | 实现入口 | 状态及工程边界 |
| --- | --- | --- |
| A01 | `--kernel vnni`；`lab/kernels.c` | C；VEX AVX-VNNI，A+128，按 packed 段扣除列补偿，int32 段界检查 |
| A02 | `--kernel avx2-sign` | C；abs(A)/sign(B,A)，±99 下证明 maddubs 不饱和；不支持完整 int8 域 |
| A03 | `--kernel nibble` | C；u8 A 拆高低四位，两个窄位宽点积后精确合成 |
| A04 | `--kernel fp32`、`blas-fp32` | C；最多 1024 个 K 项，从零计算，先转 int64 再合并；顺序 BLAS 也已实际验证 |
| A05 | `--kernel fp64`、`blas` | C；全局绝对值界小于 2^53，避免将浮点近似当作精确整数 |
| A06 | `--kernel strassen`；`lab/strassen.c` | C；一层矩形 Strassen，奇数尺寸补零，线性组合直接生成 int16 pair pack，逐乘积消费；Python 另有有限深度递归版本 |
| A07 | `--kernel strassen-winograd`；`lab/exact.py --model strassen-winograd` | C + 模型；七乘法及 Winograd 组合，C 仅保留一个乘积矩阵，叶子按 8192 个 K 项分段 |
| A08 | `lab/exact.py --model alternative-basis` | 模型；显式可逆整数基、精确逆矩阵、贪心二项公共子表达式消除、线性形式引用计数释放；未声称找到全局最优加法/存储调度 |
| A09 | `lab/exact.py --scheme schemes/strassen_2x2x3.json` | 模型；U/V/W 系数加载和逐基系数证书，内置秩 11 的 2×2×3 示例；可接收通过整数/有理数域证书的外部方案。没有把 GF(2) 方案直接移植为整数方案，也未重新训练 AlphaTensor |
| A10 | `--kernel winograd` | C；配对内积恒等式、行/列因子和奇数尾项；使用 int64 中间界守卫 |
| A11 | `lab/exact.py --model bitserial/four-russians` | 模型；有符号 8 位 bit-plane/popcount；分组子集表，符号位权重为 -128 |
| A12 | `lab/exact.py --model crt/finite-field-fp64` | 模型；CRT 的模数乘积大于两倍结果界；有限域版本每个 FP64 分段绝对值小于 2^52，转换和重构计入耗时 |
| A13 | `lab/exact.py --model structured` | 模型；重复 A 行缓存、B 的 [-7,7] 主体与所有非零残差；随机稠密输入仍精确，可能更慢 |
| A14 | `lab/exact.py --model kronecker/ntt` | 模型；大整数编码的系数位宽防进位，或三个可用 NTT 素数下的卷积与有符号 CRT；模数不足明确拒绝。没有采用未经误差证明的复数 FFT 四舍五入 |

## 本地组织 K01–K10

| ID | 实现入口 | 状态及工程边界 |
| --- | --- | --- |
| K01 | `--beta0 1/0` | C；rows 的首贡献覆盖 D 或先清零再累加；不依赖 malloc 初值。split-K/网格的首贡献也覆盖 |
| K02 | `--stream 1 --fused 1` | C；`input_pack_a` 从行解析结果直接写 quartet，补齐预写正确零编码，不保留整块 int32 A；要求 int8 核 |
| K03 | `--layout rows/interleaved` | C；按 MR 行组交错 A，同一 K-group 的行广播源相邻 |
| K04 | `--pad 0/64/128` | C；A 的 int32 输入/packed stride 和常规行分解 D 物理步长 padding，逻辑尺寸不变 |
| K05 | `--mr`、`--nr`；`tools/tune.py` | C + 工具；四种专用形状，明确拒绝其他组合；是否 spill 需结合反汇编和实测筛选 |
| K06 | `--kb`、`--rows`；`tools/tune.py` | C + 工具；多 stripe 双缓冲和行超级块；搜索空间默认 48 个候选加基线 |
| K07 | `--unroll 1/2/4`、`--pipeline 1` | C；显式展开步骤及独立的下一组 B 预加载流水，尾部不越界；流水启用时不叠加展开开关 |
| K08 | `--stream 1` | C；完整 B 就绪后逐行组解析、计算、加 C、格式化，复用小 A/D；文本缓冲最终整体定位写出 |
| K09 | `--kernel recursive --leaf N`；`lab/exact.py --model morton` | C + 模型；C 递归切最大维、标量叶子；Python 真正将 tile 转为 Morton 顺序后计算。Morton 尚未接入优化 C 微核 |
| K10 | `--kernel blas/blas-fp32/external`；`backends/` | BLAS 与 reference 插件已验证；oneDNN/FBGEMM/LIBXSMM 源码适配器为**外部依赖待验收**。没有用 reference 通过来冒充三个库通过 |

## 输入 P01–P06

| ID | 实现入口 | 状态及工程边界 |
| --- | --- | --- |
| P01 | `--parser simd` | C；256 项 shuffle 表压缩 int16 候选；自测穷举全部 mask |
| P02 | `--parser wide` | C；真正的 32 字节分类，两个 16 候选转换组，跨半区负号修复；完整 token 与行末检查，长数回退 |
| P03 | `--parser swar` | C；两位十进制的宽标量加载，加载前验证长度和结束分隔符 |
| P04 | `lab/fused_parse.py` | MPI 模型；字节分片只解析其 token 起点，延伸最后 token，prefix count 定位，owner 交换；未替换 C 的严格双遍输入路径 |
| P05 | `--prefault none/local/all` | C；本 rank 扫描片、实际行区间的页对齐建议；Linux 不支持 POPULATE_READ 时保持正常需求缺页 |
| P06 | `--reader pread/mmap` | C；pread 到私有对齐缓冲，处理 EINTR/短读；读取和映射均计时 |

## MPI M01–M13

| ID | 实现入口 | 状态及工程边界 |
| --- | --- | --- |
| M01 | `--bmode packed` | C；发送方一次 pack，rank-major packed 段附列和；接收端逐段直接消费，独立 counts/displacements 生命周期 |
| M02 | `--bmode equal` | C；ceil(K/p) 等长零填充分片 Iallgather，任意 p；这是原始 int32 B 的等长对照，未把 equal 与 packed 合并成一个模式 |
| M03 | `--bmode replicate` | C；各 rank 私有读取完整所需 B，不发 B collective |
| M04 | `--bmode root`；stream 的 `root-packed` | C；root 解析广播；root-packed 广播最终 packed B 和列和，其他 rank 不重打包 |
| M05 | `--dist ring` | C；Isend/Irecv 双缓冲，段 owner 轮转，完成请求后才复用缓冲 |
| M06 | `--dist split-k` | C；私有 K 片完整 M×N 部分结果，MPI_INT64_T Reduce_scatter，owner 只加一次 C |
| M07 | `--dist columns` | C；1×p 输出列网格，计算后 Alltoallv 转为连续行 owner，重排计入总时间 |
| M08 | `--dist summa`；MPI 模型的 `cannon/fox/hsumma` | C + 模型；矩形 SUMMA 网格；Cannon 真正初始错位/循环移位，Fox 行广播/列轮转，HSUMMA 两级广播。Python 输入目前每 rank 完整读取 |
| M09 | MPI 模型 `layered-summa --layers L` | 模型；K 按层分配、层内面板广播、fiber Reduce；支持非方形层内网格，层数必须整除 p |
| M10 | MPI 模型 `recursive/grid-search` | 模型；最大维递归划分和内存模型约束的网格/复制选择；明确为 CARMA/COSMA 思路原型，没有复刻其通信最优库实现 |
| M11 | `--progress 0/1 --progress-us 0/50/100/200` | C；packed 行组边界 MPI_Test，可加时间间隔。其他核在一次 gemm 完成后检查；需实测等待和总时间，不能从 API 名字推断真实重叠 |
| M12 | `tools/probe.py --collectives` + bench case 的 `env` | 工具；查询安装版本的 ompi_info，候选环境变量随原始样本记录；不硬编码算法编号。当前 MPICH 环境未验证 Open MPI tuned/libnbc 策略 |
| M13 | `--balance WEIGHT` | C；按计算量加 A/C 文本字节权重切连续行块；保持按 rank 顺序的连续输出，支持空块 |

## 输出 O01–O07

| ID | 实现入口 | 状态及工程边界 |
| --- | --- | --- |
| O01 | `--format jeaiii` | C/C++；完整上游 jeaiii，不仅借用名字；保留 MIT 授权，测试 int64 极值 |
| O02 | `--format pairs/quad --static-table 0/1` | C；200B 两位表、运行时四位表/生成的只读四位表；生成器可复现 |
| O03 | `--format vector` | C；8 个 int32 的 AVX2 无符号倒数乘法除十和 ASCII 构造，INT_MIN 取绝对值用位模式；超出 int32 时回退完整 int64 格式化 |
| O04 | `--format batch/vector` | C；小批次长度/前缀和后紧凑写入，不建立全矩阵长度数组；与 stream 可组合 |
| O05 | `--offset exscan/allgather` | C；rank 0 明确把 Exscan 未定义接收值设为 0，`#D` 只计入一次 |
| O06 | `--writer pwrite/mpi/mpi-all` | C；显式偏移和 MPI_File_write_at/all 对照，零输出 rank 仍参与 collective |
| O07 | `--write-chunk BYTES` | C；统一定长 ftruncate，再比较分块 pwrite；循环处理短写/EINTR/零进展；未重新引入调研中已失败的 fallocate 默认策略 |

## 工程 E01–E05

| ID | 实现入口 | 状态及工程边界 |
| --- | --- | --- |
| E01 | 默认 64B 对齐；`--aligned 0/1` | C；微核 B 的 loadu/aligned 合法对照，packed 段布局保证 32B 对齐；容量和清零独立管理 |
| E02 | `--alloc aligned/huge --memory-stats 1` | C；进程私有、2MiB 对齐/取整、MADV_HUGEPAGE、失败时 aligned 分配回退；报告实际 AnonHugePages 快照和缺页，不假定提示必生效 |
| E03 | `--arena 0/1` | C；8MiB 私有区、对齐子分配、释放后复用及最终整体回收；在途 MPI 缓冲不回收 |
| E04 | `tools/probe.py`、`tools/bind_rank.py --cores ...`、bench 的 `--cores` | 工具；读取拓扑/cgroup，验证允许 CPU 集合和物理核唯一性，每 rank 打印实际绑定；目标 P/E 核类型仍需确认 |
| E05 | Makefile 的独立 BUILD/TARGET；`tools/pgo.py` | 工具；源码构建、可选编译器/LTO 参数、固定对象路径的 generate/train/use、缺失 profile 视为错误。PGO 工具已实际跑通，性能优劣仍需独立 A/B |

## 关键实现协议

1. **数值界先于窄化。** 行扫描得到 A/B/C 的最大绝对值，用 `__int128` 检查 `K*amax*bmax+cmax <= INT64_MAX`。FP64 另验 2^53，int8 原始偏移点积另验 `Ksegment*227*99 <= INT32_MAX`。Strassen–Winograd 线性组合最多四项，即 ±396；叶子 8192 项的绝对和小于 int32 上限。
2. **zero 与 aligned 分开。** 首贡献覆盖 D；后续 K 块才读旧 D。逻辑补齐为零，VNNI 的 A 补齐字节为 128。所有 quartet/pair 拼接先转 unsigned，再移位。
3. **请求拥有自己的参数。** 每个 stripe 单独保存 counts、displacements、数据和 request。Wait/Test 使 request 完成后才允许 mem_drop/arena 复用；环传中发送缓冲的写入与计算只读分离。
4. **逻辑结果统一为 int64。** split-K 不归约偏移后的未补偿值；每个 packed 段先补偿。C 在所有 K 贡献完成后由 owner 加一次。任意归约子树都受绝对值界约束。
5. **输出 owner 统一为连续行。** 2D/列分解通过 Alltoallv 重排。输出长度用 uint64_t，字节计数转换到 MPI int 之前检查上限；root 截断后 barrier，所有 rank 写入互不相交的区间。
6. **失败不成为性能样本。** 错误使用 MPI_Abort，benchmark 验证返回码、当前输出和 hash，保留日志；超时只终止本次创建的作业进程组。文件写入失败可能留下不完整的目标文件，必须检查退出状态。
7. **公平时间边界。** 验证、解析、打包、动态加载、JIT、内存首次触页、通信、输出都在被测程序内执行。独立参考在实验工具里生成，仅用于校验，没有供被测程序读取的预计算结果缓存。

## 仍需目标环境验收的项目

这些是明确保留的验证边界，不能用“小样例通过”代替：

- 原始 `RandMatrix.txt` 和老师 `verify_`；本次使用生成输入和独立整数参考，连接器未提供这两个大文件的可用本地内容。
- 13900K 上的 1/2/4 rank 相同 P 核集合、真实编译器/MPI/文件系统、原 v8 与新候选的完整随机 A/B，以及 S(p)/E(p)。
- oneDNN/FBGEMM/LIBXSMM 的实际版本构建、隐藏线程、ISA、首次 JIT、极值正确性与计时。
- MPI 非阻塞通信的真实重叠、Open MPI 实际 collective 组件选择，以及不同 THP 策略的收益。
- Python 算法原型目前只证明可执行和精确；将 Morton、替代基、位串行、CRT/NTT、分层/递归通信移植为与当前 C 微核同水平的优化版本，是后续性能工程工作。
