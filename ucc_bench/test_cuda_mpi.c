#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    float *gpu_buf;
    cudaMalloc((void **)&gpu_buf, sizeof(float));
    cudaMemset(gpu_buf, 1, sizeof(float));

    MPI_Allreduce(MPI_IN_PLACE, gpu_buf, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    float result;
    cudaMemcpy(&result, gpu_buf, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %f\n", result);

    cudaFree(gpu_buf);
    MPI_Finalize();
    return 0;
}

