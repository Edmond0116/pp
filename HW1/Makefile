CC = mpicc
CXX = mpicxx
CFLAGS = -O3 -std=gnu99 -march=native
CXXFLAGS = -O3 -std=gnu++11 -march=native

STUDENTID = 104021219
TARGETS = HW1_$(STUDENTID)_basic HW1_$(STUDENTID)_advanced

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean: 
	rm -f $(TARGETS) out*
