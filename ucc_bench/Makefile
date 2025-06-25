# === Installation paths ===
UCC_INSTALL = /home/nkodkani/ucc/build/install
UCX_INSTALL = /home/nkodkani/ucx/build/install
CUDA_HOME   = /soft/compilers/cuda/cuda-12.2.2

# === Compiler ===
CC = /home/nkodkani/mpich_ucc/build/install/bin/mpicc

# === Include and Link flags ===
CFLAGS = -g -O3 \
				 -I$(UCC_INSTALL)/include \
         -I$(UCX_INSTALL)/include \
         -I$(CUDA_HOME)/include

LDFLAGS = -L$(UCC_INSTALL)/lib -lucc \
          -L$(UCX_INSTALL)/lib -lucp -lucs -luct \
          -L$(CUDA_HOME)/lib64 -lcudart

RPATH = -Wl,-rpath,$(UCC_INSTALL)/lib \
        -Wl,-rpath,$(UCX_INSTALL)/lib \
        -Wl,-rpath,$(CUDA_HOME)/lib64

# === Source and target ===
SRC = extensive_ucc_bench.c
TARGET = ucc_allreduce_bench

# === Build rules ===
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(RPATH)

clean:
	rm -f $(TARGET)

