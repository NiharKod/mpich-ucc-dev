#!/bin/bash

MPICH_PATH=/home/nkodkani/mpich/build/install
ROCM_PATH=/soft/compilers/rocm/rocm-6.3.2/lib

${MPICH_PATH}/bin/mpiexec \
	-host amdgpu04 \
	-n 4 \
	-ppn 4 \
	-genv FI_PROVIDER=verbs \
	-genv LD_LIBRARY_PATH=${ROCM_PATH}:${LD_LIBRARY_PATH} \
	./ucc_allreduce_bench -n 1024
