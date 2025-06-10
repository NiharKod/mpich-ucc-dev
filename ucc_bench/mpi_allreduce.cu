#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CALL(call)                                                     \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            MPI_Abort(MPI_COMM_WORLD, -1);                                  \
        }                                                                   \
    } while (0)

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float h_result;
    float *d_sendbuf, *d_recvbuf;

    // Allocate device memory
    CUDA_CALL(cudaMalloc(&d_sendbuf, sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_recvbuf, sizeof(float)));

    // Initialize send buffer with rank value
    float h_val = (float)rank;
    CUDA_CALL(cudaMemcpy(d_sendbuf, &h_val, sizeof(float), cudaMemcpyHostToDevice));

    // Perform Allreduce on GPU memory
    MPI_Allreduce(d_sendbuf, d_recvbuf, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    // Copy result back to host
    CUDA_CALL(cudaMemcpy(&h_result, d_recvbuf, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Rank %d: Allreduce result = %f\n", rank, h_result);

    CUDA_CALL(cudaFree(d_sendbuf));
    CUDA_CALL(cudaFree(d_recvbuf));

    MPI_Finalize();
    return 0;
}

