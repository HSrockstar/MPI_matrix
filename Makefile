MPICC ?= mpicc
MPICXX ?= mpicxx
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wshadow -Wconversion -Wno-sign-conversion
CXXFLAGS ?= -O3 -std=c++11 -Wall -Wextra
BUILD ?= build
TARGET ?= mpi_matmul
LDFLAGS ?=
SOURCES = mpi_matmul.c lab/common.c lab/input.c lab/output.c lab/kernels.c lab/distributed.c lab/selftest.c lab/blas.c lab/external.c lab/strassen.c
OBJECTS = $(patsubst %.c,$(BUILD)/%.o,$(SOURCES)) $(BUILD)/lab/format_jeaiii.o
DATA = 作业基于MPI的基本矩阵运算的附件
.PHONY: all clean check test sanitize run1 run2 run4 verify
all: $(TARGET)
$(TARGET): $(OBJECTS)
	$(MPICXX) $(LDFLAGS) -o $@ $(OBJECTS) -ldl
$(BUILD)/%.o: %.c lab/lab.h
	@mkdir -p $(dir $@)
	$(MPICC) $(CFLAGS) -c $< -o $@
$(BUILD)/lab/output.o: lab/quad_table.h
$(BUILD)/lab/external.o: backends/backend.h
$(BUILD)/lab/format_jeaiii.o: lab/format_jeaiii.cpp lab/vendor/jeaiii/jeaiii_to_text.h
	@mkdir -p $(dir $@)
	$(MPICXX) $(CXXFLAGS) -c $< -o $@
check: mpi_matmul
	./mpi_matmul --selftest
test: mpi_matmul
	python3 tests/integration.py
sanitize:
	$(MAKE) BUILD=build-sanitize TARGET=mpi_matmul_sanitize CFLAGS='-fno-pie -O1 -g -std=c11 -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer' CXXFLAGS='-fno-pie -O1 -g -std=c++11 -fsanitize=address,undefined -fno-omit-frame-pointer' LDFLAGS='-no-pie -fsanitize=address,undefined'
run1 run2 run4: mpi_matmul
	mpirun -np $(subst run,,$@) ./mpi_matmul '$(DATA)/RandMatrix.txt' result.txt
verify:
	'$(DATA)/verify_' '$(DATA)/RandMatrix.txt' result.txt
clean:
	rm -rf $(BUILD)
	rm -f $(TARGET)
