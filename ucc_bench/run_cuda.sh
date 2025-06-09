#!/bin/bash

# Set paths
MPICH_PATH=/home/nkodkani/mpich/build/install
CUDA_PATH=/soft/compilers/cuda/cuda-12.3.0/lib64

# Run UCC + CUDA Allreduce benchmark on gpu06
${MPICH_PATH}/bin/mpiexec \
	-host gpu06 \
	-n 4 \
	-ppn 4 \
	-genv FI_PROVIDER=verbs \
	-genv LD_LIBRARY_PATH=${CUDA_PATH}:${LD_LIBRARY_PATH} \
	./ucc_allreduce_bench -n 1024
