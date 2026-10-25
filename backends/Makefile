CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fPIC
# Prefix must contain headers and libraries for the selected, recorded version.
PREFIX ?= /usr/local
CPPFLAGS += -I$(PREFIX)/include
LDFLAGS += -shared -L$(PREFIX)/lib -L$(PREFIX)/lib64
.PHONY: reference onednn fbgemm libxsmm clean
reference onednn fbgemm libxsmm: %: libmatrix_%.so
libmatrix_reference.so: adapters.cpp backend.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -DMATRIX_REFERENCE $< $(LDFLAGS) -o $@
libmatrix_onednn.so: adapters.cpp backend.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -DMATRIX_ONEDNN $< $(LDFLAGS) -ldnnl -o $@
libmatrix_fbgemm.so: adapters.cpp backend.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -DMATRIX_FBGEMM $< $(LDFLAGS) -lfbgemm -o $@
libmatrix_libxsmm.so: adapters.cpp backend.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -DMATRIX_LIBXSMM $< $(LDFLAGS) -lxsmm -lm -o $@
clean:
	rm -f libmatrix_reference.so libmatrix_onednn.so libmatrix_fbgemm.so libmatrix_libxsmm.so
