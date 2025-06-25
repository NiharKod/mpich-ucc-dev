#!/bin/bash

# Set paths
MPICH_PATH=/home/nkodkani/mpich_ucc/build/install
CUDA_PATH=/soft/compilers/cuda/cuda-12.2.0/lib64

# Run UCC + CUDA Allreduce benchmark on gpu07
${MPICH_PATH}/bin/mpiexec \
	-host gpu07 \
	-n 4 \
	-ppn 4 \
	-genv FI_PROVIDER=verbs \
	-genv MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl \
	-genv MPIR_CVAR_ALLREDUCE_CCL=ucc \
	-genv MPIR_CVAR_DEVICE_COLLECTIVES=none \
	-genv MPIR_CVAR_CH4_OFI_ENABLE_HMEM=0 \
	-genv LD_LIBRARY_PATH=${CUDA_PATH}:${LD_LIBRARY_PATH} \
	./ucc_allreduce_bench

#	-genv MPIR_CVAR_ALLREDUCE_INTRA_ALGORITHM=ccl \
#	-genv MPIR_CVAR_ALLREDUCE_CCL=ucc \
#	-genv MPIR_CVAR_DEVICE_COLLECTIVES=none \
