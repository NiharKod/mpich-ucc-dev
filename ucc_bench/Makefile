# Change these to your actual install paths
UCC_INSTALL = /home/nkodkani/ucc/build/install
UCX_INSTALL = /home/nkodkani/ucx/build/install
ROCM_LIB    = /soft/compilers/rocm/rocm-6.3.2/lib

CC = /home/nkodkani/mpich/build/install/bin/mpicc
CFLAGS = -I$(UCC_INSTALL)/include -I$(UCX_INSTALL)/include
LDFLAGS = -L$(UCC_INSTALL)/lib -lucc \
          -L$(UCX_INSTALL)/lib -lucp -lucs -luct \
          -L$(ROCM_LIB)
RPATH = -Wl,-rpath,$(UCC_INSTALL)/lib \
        -Wl,-rpath,$(UCX_INSTALL)/lib \
        -Wl,-rpath,$(ROCM_LIB)

SRC = ucc_allreduce_bench.c
TARGET = ucc_allreduce_bench

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(RPATH)

clean:
	rm -f $(TARGET)

