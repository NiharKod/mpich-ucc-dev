#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>   // for getpid(), sleep()
#include <stdio.h>    // for printf(), fflush()

volatile int i = 0;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
//    int dummy;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("Rank %d PID %d on %s ready for attach\n", rank, getpid(), hostname);
    fflush(stdout);
    while (i == 0)
        sleep(5);


    int *sendbuf, *recvbuf;
    cudaMalloc((void **)&sendbuf, sizeof(int));
    cudaMalloc((void **)&recvbuf, sizeof(int));

    int val = rank;
    cudaMemcpy(sendbuf, &val, sizeof(int), cudaMemcpyHostToDevice);

    MPI_Allreduce(sendbuf, recvbuf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    int result;
    cudaMemcpy(&result, recvbuf, sizeof(int), cudaMemcpyDeviceToHost);
    if (rank == 0) {
        printf("Allreduce result: %d\n", result);
    }

    cudaFree(sendbuf);
    cudaFree(recvbuf);
    MPI_Finalize();
    return 0;
}

