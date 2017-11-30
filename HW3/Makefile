CC = mpicc
CXX = mpicxx
LDFLAGS = -pthread
CFLAGS = -O3 -std=gnu99 -march=native
CXXFLAGS = -O3 -std=gnu++11 -march=native
TARGETS = APSP_Pthread APSP_MPI_sync APSP_MPI_async APSP_Hybrid

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o) *.out
