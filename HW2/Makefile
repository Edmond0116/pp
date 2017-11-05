CC = mpicc
CXX = mpicxx
LDFLAGS = -lpng
CFLAGS = -fopenmp -O3 -std=gnu99 -march=native
CXXFLAGS = -fopenmp -O3 -std=gnu++11 -march=native
TARGETS = ms_mpi_static ms_mpi_dynamic ms_omp ms_hybrid

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
