CC      = mpicc
CCBIN   = gcc-12
CFLAGS  = -O3 -Wall -Wextra -march=native -mavxvnni
DATA    = 作业基于MPI的基本矩阵运算的附件

# 最终提交：直接 make 即可（无 PGO）。gcc-12：gcc-10 不识别 -mavxvnni，
# 无 VNNI 的机器自动回退 vpmaddwd+vpaddd）
all: mpi_matmul

mpi_matmul: mpi_matmul.c
	OMPI_CC=$(CCBIN) $(CC) $(CFLAGS) -o $@ $<

# PGO（可选，实测中性偏负，不推荐）：v8 修复了旧 Makefile "一步编译链接导致
# gcda 名字错位、PGO 静默失效"的坑后，真正的 PGO 首次跑通——交错实测
# np=1 −2% / np=2 0% / np=4 +5.6%，弃用（旧日志的 +3~5% 系无效 PGO 的误判）。
# 若要实验：注意必须两步编译（-c 出固定名 .o），gcda 与 .o 同名同目录。
pgo: mpi_matmul.c
	OMPI_CC=$(CCBIN) $(CC) $(CFLAGS) -fprofile-generate -c -o mpi_matmul.o mpi_matmul.c
	OMPI_CC=$(CCBIN) $(CC) -fprofile-generate -no-pie mpi_matmul.o -o mpi_matmul_pgo_train
	mpirun --bind-to core -np 1 ./mpi_matmul_pgo_train "$(DATA)/RandMatrix.txt" /tmp/r_pgo_train.txt
	mpirun --bind-to core -np 2 ./mpi_matmul_pgo_train "$(DATA)/RandMatrix.txt" /tmp/r_pgo_train.txt
	mpirun --bind-to core -np 4 ./mpi_matmul_pgo_train "$(DATA)/RandMatrix.txt" /tmp/r_pgo_train.txt
	OMPI_CC=$(CCBIN) $(CC) $(CFLAGS) -fprofile-use -fprofile-correction -c -o mpi_matmul.o mpi_matmul.c
	OMPI_CC=$(CCBIN) $(CC) -no-pie mpi_matmul.o -o mpi_matmul
	rm -f mpi_matmul_pgo_train mpi_matmul.gcda

# 单进程（串行基线，用于加速比 T1）
run1: mpi_matmul
	mpirun --bind-to core -np 1 ./mpi_matmul "$(DATA)/RandMatrix.txt" result.txt

run2: mpi_matmul
	mpirun --bind-to core -np 2 ./mpi_matmul "$(DATA)/RandMatrix.txt" result.txt

run4: mpi_matmul
	mpirun --bind-to core -np 4 ./mpi_matmul "$(DATA)/RandMatrix.txt" result.txt

# 用老师提供的 verify_ 验证 result.txt
verify:
	"$(DATA)/verify_" "$(DATA)/RandMatrix.txt" result.txt

# 交错 A/B 跑分（共享服务器必须同窗口交错 + ≥10 轮取中位数）
bench_ab: bench_ab.sh
	@echo '用法: ./bench_ab.sh <binA> <binB> [np列表] [轮数]'

clean:
	rm -f mpi_matmul result.txt

.PHONY: all run1 run2 run4 verify pgo bench_ab clean
