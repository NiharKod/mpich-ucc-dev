#include <stdio.h>
#include <mpi.h>
#include "mpi-ext.h" // Needed for CUDA-aware check

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    printf("Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library has CUDA-aware support.\n");
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
    printf("This MPI library does NOT have CUDA-aware support.\n");
#else
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif

    printf("Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
    if (1 == MPIX_Query_cuda_support()) {
        printf("This MPI library has CUDA-aware support.\n");
    } else {
        printf("This MPI library does NOT have CUDA-aware support.\n");
    }
#else
    printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif

    MPI_Finalize();
    return 0;
}

