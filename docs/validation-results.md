# 实现验证记录

日期：2026-10-25。对应源码以本目录 `source-manifest.json` 的逐文件 SHA-256 为准；这里记录正确性与工具流程验收，不宣称目标服务器上的加速比。

## 环境与范围

- Linux x86-64，Intel Xeon Platinum 8573C，具有 AVX2/AVX-VNNI；不是项目原始 13900K。
- GCC/G++ 13.3.0、MPICH 4.3.2、Python 3.12、NumPy 2.3.5、mpi4py 4.1.2。外部 BLAS 为系统顺序参考 CBLAS LP64。
- 发布构建使用 `-O3 -std=c11 -Wall -Wextra -Wshadow -Wconversion -Wno-sign-conversion -Werror`，C++ wrapper 使用 `-O3 -std=c++11 -Wall -Wextra -Werror`。没有全局 `-march=native`/AVX 标志。
- ASan/UBSan 使用独立对象目录和二进制，`-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`。本环境以 non-PIE 构建 sanitizer 版本。`detect_leaks=0`：LeakSanitizer 的进程检查在该受限环境失败，不能宣称完成了泄漏检测。
- 运行时使用 `MPIR_CVAR_CH4_NETMOD=ofi FI_PROVIDER=tcp`。最初尝试的 OFI shm 在重复启动后发生 MPI_Init 资源冲突，发生在应用入口之前；改用 tcp 后完成检查，没有将那些失败启动计为通过。此前 MPICH 5.0.2 的环境启动问题也未当作程序测试结果。
- 未取得原始大输入和 `verify_` 的可执行本地内容，因此没有声称通过老师验证器。原始尺寸测试使用生成的 ±99 输入及独立 NumPy int64 矩阵乘法。

## 已执行检查

| 检查 | 结果 | 覆盖 |
| --- | --- | --- |
| 严格编译 | 通过，零 warning | 所有默认 C/C++ 翻译单元，`-Werror` |
| `--selftest`，4 rank | 每 rank 1386 项通过 | 256 mask；384 个解析组合；1 个 packed B lane/补零/列和验证；660 个核/布局/值域组合；85 个 int64 格式化边界 |
| Release 端到端 | 221 个配置通过 | np=1/2/3/4，五种分解，多 stripe、B 模式、微核形状、展开/流水、parser/formatter/writer、fused stream、arena、Python MPI 原型 |
| Release 内存污染 | 同一 221 配置通过 | `MALLOC_PERTURB_=165`，覆盖首次 D 覆盖写、输入补齐、arena 复用 |
| ASan + UBSan 端到端 | 221 个配置通过 | 与 release 相同的集成配置，错误立即终止 |
| ASan + UBSan 边界 | 156 个配置通过 | 随机空格/tab/CRLF/无最终换行，80 组解析端到端组合，int32 输入极值/int64 输出，多余/缺少值、非法 token、溢出拒绝，以及大 K 的空行 owner |
| POSIX 写故障注入 | 3 项通过 | LD_PRELOAD 注入 EINTR+7 字节短写时结果完整；zero progress、ENOSPC 均失败退出 |
| 完整作业尺寸 | 9 个配置逐元素相等 | 1023×2047×1023；np=1/2/4 VNNI、packed AVX2、np=3 split-K、fused stream、SUMMA FP32 BLAS、Strassen、Strassen–Winograd |
| Python 精确模型 | 6 个 unittest 测试组通过 | 13 个模型×5 种形状×3 种值分布，共 195 次模型/参考比较，另有大整数、符号位、坏证书、矩形证书及 NTT 拒绝测试 |
| 外部插件 ABI | reference 后端 36 项通过 | m/k=0、odd K、K=2047、正负极值、stride、add=0/1、输出 padding 保持 |
| A/B 测量工具 | 流程通过 | 4 个配置、随机顺序、每次独立输出/参考比较、每 rank 物理核绑定及原始样本保存；短 smoke 样本不用于宣布加速 |
| 调参生成器 | 通过 | 49 个有效候选，MR 与行组整除关系正确 |
| PGO | generate → np=1/2/4 train → use → 精确验证通过 | 同一路径对象文件，确实生成 gcda，`missing-profile`/`coverage-mismatch` 作为错误 |
| 反汇编 | 通过 ISA 检查 | 检出的 640 条 `vpdpbusd` 均为 VEX 编码，未检出 EVEX `vpdpbusd`；见 `assembly-review.json` |
| oneDNN/FBGEMM/LIBXSMM | **未执行** | 本环境未安装；仅提供真实适配器源码及测试入口，不能外推 reference 后端的结果 |

完整尺寸生成输入 SHA-256 为 `78d9dccc58c8bc5385ea2aef23ae2e47eaaf1ec835d7b0b98d34c49c5bdadcce`。`full_size.py` 的 `validation_wall_s` 包含输出读取和比对，不能当作被测程序运行时间或性能排名；程序自身的 JSON 与外部 benchmark 分别提供正式计时入口。

反汇编中部分函数存在较多栈引用，展开/流水与大 MR/NR 可能带来寄存器压力。栈引用计数包含参数、尾部和不同分支，不能全部等同于热点 spill；这里没有宣称所有候选都达到寄存器驻留或优于基线。

## 原工程清单的验收对照

“实现/测试”与“目标实测”分开记录；后者需要实际机器和依赖，不能由静态代码审查代替。

| 编号 | 结论与证据 |
| --- | --- |
| Q01 | int32 解析与全局值域验证在窄化之前；超出专用核范围拒绝，auto 保留宽位标量路径 |
| Q02 | 每行 end、精确元素数、标记与总行数；异常输入测试通过 |
| Q03 | 空格/tab/负号/CRLF/无末尾换行已测；不支持的 token 报错 |
| Q04 | 16/32 字节路径分别要求 17/33 字节可读；尾部回退，ASan 与行边界测试通过 |
| Q05 | 全 256 mask 穷举，96 种行长和随机分隔符/跨块 token 测试 |
| Q06 | fused pack 保留只读原文和行偏移；先验证，再生成专用布局 |
| Q07 | 小数据 Python 任意精度参考，完整尺寸 NumPy int64 并先证明范围；守卫使用 __int128 |
| Q08 | 普通点积/偏移点积/补偿/C/FP32/FP64/Strassen 中间界均在实现中分开处理 |
| Q09 | VNNI、abs/sign、nibble 极值/随机及独立整数参考比较通过 |
| Q10 | 专用核拒绝不支持的值域；模型测试 -128，通用输出测试 INT32_MIN/INT64_MIN 等边界 |
| Q11 | FP32 分段从零、先转 int64；顺序 CBLAS 路径实测通过；其他库仍待验证 |
| Q12 | 双线性 U/V/W 逐基证书及坏证书拒绝测试；基变换使用 Fraction |
| Q13 | pack 元数据记录逻辑 K/N、k4、panels/bytes；不同 leading dimension 测试 |
| Q14 | 首贡献覆盖写；beta0=0 预清零；污染分配和 sanitizer 通过 |
| Q15 | quartet/pair 及 M/N 尾部零编码显式写入；column sum 不含虚构非零值 |
| Q16 | index-coded packed B 逐 lane/列和检查；跨 rank 段和奇数边界端到端比较 |
| Q17 | 四种编译期微核形状，CLI 拒绝错误 MR/NR；所有形状均测 |
| Q18 | 64B/2MiB 对齐、容量、清零独立；污染内存通过。mmap 失败回退有代码，未注入该失败 |
| Q19 | unsigned 拼接、dlsym 用 memcpy 转换函数指针；UBSan 未发现问题 |
| Q20 | B 的 aligned load 根据分配与 packed 段步长证明；格式化只写实际分配范围 |
| Q21 | 新版没有手写内联汇编；使用编译器 intrinsic/目标函数，外部 C++ ABI 捕获异常 |
| Q22 | objdump 核对 VEX VNNI，保留栈引用诊断；未在目标机器完成端口压力/llvm-mca 性能分析 |
| Q23 | baseline 翻译单元无全局 AVX，运行时 AVX2 与 CPUID 检查；实际测试机器仅一种 ISA 集合 |
| Q24 | ASan/UBSan 全套及异常边界通过；LeakSanitizer 未完成，见环境说明 |
| Q25 | 双 stripe 独立 request/counts/displacements；Wait/Test 完成前不回收 |
| Q26 | 一致 collective 顺序；格式/值域错误通过 MPI_Abort 失败，测试有超时保护 |
| Q27 | allgather 原地发送位置及 packed byte displacement 已做 np=1/2/3/4 比较 |
| Q28 | int32 B、byte packed payload、int64 结果分别采用明确单位，不依赖派生 datatype |
| Q29 | MPI_INT32_T/MPI_INT64_T/MPI_UINT64_T；count 转 int 前检查，文件偏移 64 位 |
| Q30 | 非整除、尾 stripe、空 K/空行 owner 已测；Cannon/Fox 非平方进程数明确拒绝 |
| Q31 | 非阻塞与 MPI_Test 代码已实现；实际后台推进与通信/计算重叠收益**待目标时间线验证** |
| Q32 | Reduce_scatter 的任意子树受绝对值界保护；补偿和 C 仅计一次，极值比较通过 |
| Q33 | 网格重分布及各 MPI 原型比较通过；递归原型不声称达到通信最优 |
| Q34 | stream 仅允许完整 B 模式；完成全 K 后才加 C/格式化，不混用 stripe 调度 |
| Q35 | 主程序固定线程环境；已验证顺序 BLAS。三个可选库的线程/JIT 行为**待实际依赖验证** |
| Q36 | 查询工具记录安装版本；当前无 Open MPI，tuned/libnbc 实际选路**待目标验证** |
| Q37 | 85 个 int64 格式化边界及完整输出比较；vector 超过 int32 时安全回退 |
| Q38 | size_t 乘法溢出检查、每数 32 字节容量；MPI-IO 超 int count 会明确拒绝 |
| Q39 | rank 0 唯一 #D；offset/重分布/零输出 rank 在 pwrite/MPI-IO 下验证 |
| Q40 | 输入/输出路径和 inode 检查；root truncate 后 barrier；同 inode 拒绝测试 |
| Q41 | EINTR/短写/零进展/ENOSPC 注入通过；open/write/close 状态检查 |
| Q42 | 明确无 fsync；MPI-IO 完成/关闭后才释放缓冲。未测持久介质落盘性能 |
| Q43 | probe 与 bind_rank 记录 CPU/允许集合/SMT/cgroup/绑定；原 13900K 拓扑尚未取得 |
| Q44 | 绑定脚本拒绝相同物理核的两个逻辑 CPU；当前 4-rank smoke 已运行，目标 P 核集合待确认 |
| Q45 | 分配、触页、madvise 都计时；smaps 仅在 memory-stats 诊断开关下执行 |
| Q46 | 输出实际大页快照和缺页计数；未以 madvise 成功作为 THP 生效证明，故障回退未注入 |
| Q47 | arena 释放后复用、活跃块不复用、双释放/非法释放检查、最终元数据清理；sanitizer 通过 |
| Q48 | 只改变本进程/本次作业设置；没有更改全局内核参数、清空系统缓存或终止其他作业 |
| Q49 | 内部/进程计时 9 位小数；benchmark 使用 perf_counter_ns 保存外部 wall time |
| Q50 | 当前为三个明确的复合阶段，stream 单独标记；**尚无独立 parse/pack/wait/format/write 细粒度剖析**，不将复合桶当纯算术时间 |
| Q51 | 同预热策略、每轮随机候选顺序、按轮配对；bench 保存全部有效样本 |
| Q52 | 每轮唯一输出，失败/错误 hash 即停止，保存 stdout/stderr/返回码；作业超时清理仅限自身进程组 |
| Q53 | 中位数/IQR/配对 bootstrap 已实现；本次短工具验收数据不作为加速结论 |
| Q54 | 每次从输入重新计算，统一无 fsync 语义；启动成本单独由外部计时捕获 |
| Q55 | 源码清单/二进制 SHA/编译日志/MPI 版本已记录；移除根目录旧构建产物，历史 v8 源码保留 |
| Q56 | 实际运行 PGO 三阶段，固定对象路径，缺失/不匹配 profile 使构建失败 |
| Q57 | 独立整数参考已完成；老师 verify_ 与原始输入**未能执行** |
| Q58 | 按文件和实验 ID 组织，开关支持独立对照，整体提交可回退；未替所有组合预先宣称 A/B 收益 |
| Q59 | jeaiii MIT 文件保留；默认依赖/可选库/ISA 限制已说明；外部库版本仍需在目标机器固定 |
| Q60 | **待目标机器复测**同算法 np=1/2/4 的时间、S(p)、E(p)，本次不发布跨机器推导的加速比 |

## 可复现命令与发布边界

具体命令见根目录 README。当前测试环境额外设置 OFI/tcp 和本地 MPI 动态库路径；这些是本环境的启动配置，不应无条件复制到用户的 Open MPI 环境。

仓库提交源码、实验配置、文档、许可证和精简验证记录，不提交本机二进制、PGO profile、生成的大输入、结果矩阵或工具安装目录。原有历史版本继续保留。三个可选库、真实硬件速度和老师验收是后续明确需要完成的环境验证，不能把这份记录理解为它们已经通过。
