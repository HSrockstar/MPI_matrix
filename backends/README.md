# 可选外部计算库适配器

`adapters.cpp` 提供 oneDNN、FBGEMM、LIBXSMM 三个实际适配器，以及验证插件 ABI 的 reference 后端。C/MPI 主程序通过 `--kernel external` 和 `MATMUL_BACKEND_LIBRARY` 加载独立 `.so`，默认构建不依赖它们。

**本次仅 reference 后端在实际编译/运行环境完成验证；三个外部库未安装，尚未完成各自的编译和运行验收。** 源码依据下面的官方 API 编写。它们不能算作已经验证的加速方案；在目标环境通过契约测试、老师验证器和完整计时后再启用。

```sh
make -C backends reference
python3 tests/test_backend.py backends/libmatrix_reference.so
MATMUL_BACKEND_LIBRARY="$PWD/backends/libmatrix_reference.so" \
  mpiexec -n 4 ./mpi_matmul input.txt result.txt --kernel external

# PREFIX 指向自己安装、版本已记录的库；按需设置 LD_LIBRARY_PATH。
make -C backends onednn PREFIX=/path/to/onednn
make -C backends fbgemm PREFIX=/path/to/fbgemm
make -C backends libxsmm PREFIX=/path/to/libxsmm
python3 tests/test_backend.py backends/libmatrix_onednn.so
python3 tests/test_backend.py backends/libmatrix_fbgemm.so
python3 tests/test_backend.py backends/libmatrix_libxsmm.so
```

oneDNN 必须用 `ONEDNN_CPU_RUNTIME=SEQ` 构建，适配器在头文件配置不符时拒绝编译。FBGEMM 以 `thread_id=0, num_threads=1` 调用，没有 OpenMP 区域。LIBXSMM 使用 FP64 的 64×64 tile 和运行时生成核；首次 dispatch/JIT、转换、打包及释放都处于计时范围。

oneDNN/FBGEMM 使用 `A+128` 的低/高四位各一次 u4×s8 点积，按 K=512 分段，以 int64 合成 `low + 16*high - 128*sum(B)`。每个 maddubs 相邻乘积和绝对值不超过 `2*15*99=2970`，每段累加不超过 `512*15*99=760320`。这样同时避开 AVX2 的 int16 饱和及潜在 FP32 epilogue 的整数舍入风险；没有量化输出或浮点补偿。仍须以实际库版本验证，包括极值、奇数 K、stride 和首次初始化成本。

ABI 只承诺输入 A/B 在 ±99、C 由调用方在归约后添加、D 为 int64、add=0 覆盖/add=1 累加。所有 C++ 异常在边界内转成错误码，主程序通过 MPI_Abort 一致失败。动态库是用户选择的可信本地代码；文件缺失、符号缺失和运行错误不会静默回退。

顺序 BLAS 另有无需构建插件的 `--kernel blas` 和 `--kernel blas-fp32`，默认加载 `libblas.so.3`，可以用 `MATMUL_BLAS_LIBRARY` 指定。要求 CBLAS LP64 ABI；代码拒绝可检测的 ILP64 OpenBLAS/MKL 配置。未知实现的 ABI 必须由使用者确认，不能自动从通用符号证明。主程序将 OMP/OpenBLAS/MKL 线程数设为 1，并检查可查询的 OpenBLAS 线程数。本次用系统顺序参考 BLAS 验证了这两个入口。

官方接口依据：

- [oneDNN matmul C++ 示例](https://github.com/uxlfoundation/oneDNN/blob/main/examples/primitives/matmul.cpp)，检查时文件 blob 为 `bfafc463acd59a038fa428e80afb036b47a9884a`。
- [FBGEMM PackA/PackB、memCopy、fbgemmPacked 测试](https://github.com/pytorch/FBGEMM/blob/main/test/PackedRequantizeTest.cc) 与 [头文件](https://github.com/pytorch/FBGEMM/blob/main/include/fbgemm/Fbgemm.h)。
- [LIBXSMM dispatch 示例](https://github.com/libxsmm/libxsmm/blob/main/samples/hello/hello.c)，检查时 blob 为 `8c2e98142fe0dede08816f7cc91315db0fb800d2`。

这里仅调用这些库的接口，没有复制其实现。jeaiii 是单独引入的上游头文件，许可证见 `lab/vendor/jeaiii/LICENSE`。
