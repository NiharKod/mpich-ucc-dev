#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    float *sendbuf, *recvbuf;
    cudaMalloc(&sendbuf, sizeof(float));
    cudaMalloc(&recvbuf, sizeof(float));

    float val = (float)rank;
    cudaMemcpy(sendbuf, &val, sizeof(float), cudaMemcpyHostToDevice);

    // Native MPICH will detect this is a GPU pointer if built with CUDA-aware support
    MPI_Allreduce(sendbuf, recvbuf, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    float result;
    cudaMemcpy(&result, recvbuf, sizeof(float), cudaMemcpyDeviceToHost);
    if (rank == 0) {
        printf("Allreduce result: %f\n", result);
    }

    cudaFree(sendbuf);
    cudaFree(recvbuf);
    MPI_Finalize();
    return 0;
}

