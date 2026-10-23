CC      = mpicc
CFLAGS  = -O3 -Wall -Wextra -march=native
DATA    = 作业基于MPI的基本矩阵运算的附件

all: mpi_matmul

mpi_matmul: mpi_matmul.c
	$(CC) $(CFLAGS) -o $@ $<

# PGO 构建（推荐提交用）：先训练采样再按 profile 编译，实测再快 3~5%
pgo: mpi_matmul.c
	mpicc $(CFLAGS) -fprofile-generate -o mpi_matmul_pgo_train mpi_matmul.c
	mpirun --bind-to core -np 1 ./mpi_matmul_pgo_train "$(DATA)/RandMatrix.txt" /tmp/r_pgo_train.txt
	mpirun --bind-to core -np 2 ./mpi_matmul_pgo_train "$(DATA)/RandMatrix.txt" /tmp/r_pgo_train.txt
	mpirun --bind-to core -np 4 ./mpi_matmul_pgo_train "$(DATA)/RandMatrix.txt" /tmp/r_pgo_train.txt
	mpicc $(CFLAGS) -fprofile-use -fprofile-correction -o mpi_matmul mpi_matmul.c
	rm -f mpi_matmul_pgo_train mpi_matmul.gcda

# 单进程（串行基线，用于加速比 T1）
run1: mpi_matmul
	mpirun -np 1 ./mpi_matmul "$(DATA)/RandMatrix.txt" result.txt

run2: mpi_matmul
	mpirun -np 2 ./mpi_matmul "$(DATA)/RandMatrix.txt" result.txt

run4: mpi_matmul
	mpirun -np 4 ./mpi_matmul "$(DATA)/RandMatrix.txt" result.txt

# 用老师提供的 verify_ 验证 result.txt
verify:
	"$(DATA)/verify_" "$(DATA)/RandMatrix.txt" result.txt

clean:
	rm -f mpi_matmul result.txt

.PHONY: all run1 run2 run4 verify pgo clean
