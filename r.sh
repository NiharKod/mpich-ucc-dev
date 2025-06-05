#!/bin/bash
export LD_LIBRARY_PATH=/soft/compilers/rocm/rocm-6.3.2/lib:$LD_LIBRARY_PATH
export FI_PROVIDER=tcp
/home/nkodkani/mpich/build/install/bin/mpiexec "$@"
