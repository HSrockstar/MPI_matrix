# MPI_matrix 实验实现

计算精确的整数 `D = A × B + C`，默认尺寸为 `1023 × 2047 × 1023`。新版将调研中的 55 个实验方向做成可运行的 C/MPI 路径、Python 算法原型和实验工具。每个方向的入口、限制和验证状态见 [实现清单](docs/implementation-status.md)，实际测试结果见 [验证记录](docs/validation-results.md)。

原始优化实现仍保存在 `mpi_matmul_v8.c`。新版默认配置是实验基线；本次未在原 13900K 上证明它优于 v8。根目录旧可执行文件和 `.o` 不再作为发布产物，请从源码构建。

## 构建和运行

需要 Linux、GCC/G++ 12 或更高版本、同一套 MPI 的 C/C++ wrapper。已测试 GCC 13.3、MPICH 4.3.2。默认构建不需要 BLAS、Python 或其他外部计算库；jeaiii 的头文件及 MIT 许可证已附在源码中。

```sh
make -j2
mpiexec -n 4 ./mpi_matmul \
  '作业基于MPI的基本矩阵运算的附件/RandMatrix.txt' result.txt
make verify
```

需要指定编译器时使用 `make MPICC=/path/to/mpicc MPICXX=/path/to/mpicxx`。不要使用全局 `-march=native` 来替代 ISA 分派。默认 `auto` 依次选择 AVX-VNNI、AVX2 符号变换、标量 int64 路径；显式请求不支持的 ISA 会报错。

各 rank 必须能够读取同一份完整输入；多节点运行需要共享输出文件系统。运行期间不要并发修改输入或用另一作业覆盖同一输出路径。

文件依次为 `#A`、M 行 A、`#B`、K 行 B、`#C`、M 行 C；每行列数必须正确。支持负号、空格、tab、CRLF 和最后一行无换行。输入先以 int32 验证，再按算法窄化。通用结果使用 int64，不能证明累加不溢出时拒绝输入。窄位宽/Strassen/FP32 路径要求 A/B 在 `[-99,99]` 内。

```sh
./mpi_matmul --features
./mpi_matmul --help
mpiexec -n 4 ./mpi_matmul input.txt result.txt --m 31 --k 67 --n 29 \
  --kernel vnni --parser wide --bmode packed --kb 512
mpiexec -n 4 ./mpi_matmul input.txt result.txt --m 31 --k 67 --n 29 \
  --kernel vnni --stream 1 --fused 1 --bmode root-packed \
  --parser wide --format jeaiii --arena 1
```

所有选项的数值是运行时参数。MR/NR 仅接受 `4×16`、`6×16`、`8×8`、`4×24`；`--rows` 必须是 MR 的倍数，`--pad` 必须是 64 的倍数。`--stream 1` 仅支持 rows 分解及 root/root-packed/replicate B；`--fused 1` 需要 stream 和 int8 微核。root-packed 是流式路径的选项。packed 通信需要 int8 微核。`--pipeline 1` 使用预加载流水；该模式下 `--unroll` 不参与调度。

## 主要入口

| 类型 | 选项或程序 |
| --- | --- |
| 算术核 | `--kernel auto/scalar/vnni/avx2-sign/nibble/fp32/fp64/blas/blas-fp32/winograd/strassen/strassen-winograd/recursive/external` |
| MPI 分解 | `--dist rows/ring/split-k/summa/columns` |
| rows 的 B 交换 | `--bmode allgatherv/equal/packed/root/root-packed/replicate` |
| 打包和缓存 | `--mr`、`--nr`、`--kb`、`--rows`、`--pad`、`--layout rows/interleaved`、`--beta0 0/1`、`--aligned 0/1` |
| 调度 | `--unroll 1/2/4`、`--pipeline 0/1`、`--progress 0/1`、`--progress-us N`、`--balance WEIGHT` |
| 解析和读取 | `--parser scalar/simd/swar/wide`、`--reader mmap/pread`、`--prefault none/local/all` |
| 格式化 | `--format pairs/quad/jeaiii/snprintf/batch/vector`、`--static-table 0/1` |
| 写出 | `--writer pwrite/mpi/mpi-all`、`--offset exscan/allgather`、`--write-chunk BYTES` |
| 分配 | `--alloc aligned/huge`、`--arena 0/1`、`--memory-stats 0/1` |
| 研究算法 | `lab/exact.py`：替代基、CSE、双线性证书、位串行、Four Russians、CRT、有限域 FP64、稀疏残差、Kronecker、NTT、Morton |
| MPI 原型 | `lab/mpi_models.py`：Cannon、Fox、HSUMMA、分层 SUMMA、递归划分、网格/复制因子搜索 |
| 合并扫描解析 | `lab/fused_parse.py`：按字节片解析、修复跨片 token、按 owner 交换 |

`--balance` 只适用于 rows/ring。`--bmode` 只控制 rows；split-k、summa、columns、ring 使用自己的通信协议。递归核的叶子阈值为 `--leaf`。stream 在完成全部 K 后立即格式化行组，但仍保留本 rank 的文本输出缓冲，最后统一定位写出。

## 检查

```sh
./mpi_matmul --selftest
python3 -m pip install -r requirements-lab.txt
MPIEXEC='mpiexec' python3 tests/integration.py --models
MPIEXEC='mpiexec' python3 tests/boundaries.py
MPIEXEC='mpiexec' python3 tests/io_faults.py
python3 -m unittest discover -s tests -p test_exact.py -v
MPIEXEC='mpiexec' python3 tests/full_size.py
make sanitize
ASAN_OPTIONS=detect_leaks=0:abort_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  MATMUL_BINARY="$PWD/mpi_matmul_sanitize" MPIEXEC='mpiexec' \
  python3 tests/integration.py --models
```

完整 SIMD 实验套件需要 AVX2；没有 AVX-VNNI 时自动改测 AVX2 路径。`--selftest` 可跳过不可用的 SIMD 核。`full_size.py` 明确测试 VNNI，需要该 ISA。ASan 示例关闭 LeakSanitizer，是因为本次受限运行环境无法支持其进程检查；地址和未定义行为检查仍启用。发布前应在目标环境进一步运行可用的泄漏检查及老师的 `verify_`。

## 测量和研究原型

```sh
python3 tools/probe.py --output environment.json --collectives
python3 tools/generate_fixture.py small.txt
python3 tools/bench.py small.txt --cases bench_cases.json \
  --result artifacts/ab --np 4 --trials 30 --warmup 2 --cores auto
python3 tools/tune.py small.txt --output artifacts/tune
python3 tools/pgo.py small.txt --output artifacts/pgo
python3 lab/exact.py small.txt exact.txt --model alternative-basis
python3 lab/exact.py small.txt tensor.txt --scheme schemes/strassen_2x2x3.json
mpiexec -n 4 python3 lab/mpi_models.py small.txt cannon.txt --algorithm cannon
mpiexec -n 4 python3 lab/mpi_models.py small.txt layers.txt --algorithm layered-summa --layers 2
mpiexec -n 4 python3 lab/fused_parse.py small.txt fused.txt --m 31 --k 67 --n 29
```

`bench.py` 为每次启动创建独立输出，随机安排候选顺序，使用独立参考校验每个结果，保留命令、stdout/stderr、返回码、输入/二进制 SHA-256、原始 wall time、中位数/IQR 和配对 bootstrap 区间。`--result` 必须是新目录。调参器默认只生成 49 个配置；加 `--run` 才执行，筛选胜者还需新的独立样本确认。

`--cores auto` 选择当前允许集合中不同物理核，无法自动保证混合架构上全为 P 核；13900K 应依据拓扑明确提供 CPU 列表。启动器预先限制亲和性时，需先调整其绑定选项，再使用脚本。`--memory-stats 1` 是单独诊断运行，读取 smaps 的时间会计入程序时间。

程序 JSON 的 `validation_s` 包含映射、行索引、格式和值域验证；`load_compute_s` 包含后续解析、打包、通信和算术；`output_cleanup_s` 包含格式化、写出和释放。stream 模式的格式化/写出位于 `load_compute_s`，以 `stream_output_in_compute=1` 标记。各桶分别取 rank 最大值，不能相加推导总时间。外部 wall time 包含 MPI 启动与退出。输出不执行 fsync，比较的是相同的页缓存完成语义。

Python 原型用于检验算法和通信方式，读取及输出成本包含在自身计时中。Cannon/Fox 要求平方进程数；分层算法要求 layers 整除进程数。递归/网格搜索借鉴 CARMA/COSMA 思路，未复刻其完整优化实现；`--memory` 是网格模型的局部 tile 预算，不是进程总内存限制。依赖外部库的入口和验证边界见 [适配器说明](backends/README.md)。
