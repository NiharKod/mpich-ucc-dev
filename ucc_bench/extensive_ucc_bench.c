#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define NITER 1

#define CUDA_CALL(call)                                                   \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "%s:%d CUDA error %d: %s\n", __FILE__, __LINE__, \
                    err, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define MPI_CALL(call)                                                    \
    do {                                                                  \
        int err = call;                                                   \
        if (err != MPI_SUCCESS) {                                         \
            char errstr[MPI_MAX_ERROR_STRING];                            \
            int len = 0;                                                  \
            MPI_Error_string(err, errstr, &len);                          \
            fprintf(stderr, "%s:%d MPI error: %s\n", __FILE__, __LINE__, errstr); \
            MPI_Abort(MPI_COMM_WORLD, err);                               \
        }                                                                 \
    } while (0)

int main(int argc, char** argv) {
    MPI_CALL(MPI_Init(&argc, &argv));
    int rank, size;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

    // allocate one int on device for send/recv
    int *d_send, *d_recv;
    CUDA_CALL(cudaMalloc((void **)&d_send, sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_recv, sizeof(int)));

    // list of ops to test
    MPI_Op ops[]       = { MPI_SUM,  MPI_PROD,  MPI_MIN,    MPI_MAX };
    const char *names[] = { "SUM",    "PROD",    "MIN",      "MAX" };
    const int n_ops = sizeof(ops)/sizeof(*ops);

    for (int op_i = 0; op_i < n_ops; ++op_i) {
        MPI_Op  op   = ops[op_i];
        char   *name = (char*)names[op_i];

        // set up the send value = rank+1
        int h_send = rank + 1;
        MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
        CUDA_CALL(cudaMemcpy(d_send, &h_send, sizeof(int), cudaMemcpyHostToDevice));

        // time NITER Allreduce
        double t0 = MPI_Wtime();
        for (int i = 0; i < NITER; ++i) {
            MPI_CALL(MPI_Allreduce(d_send, d_recv, 1, MPI_INT, op, MPI_COMM_WORLD));
        }
        double t1 = MPI_Wtime();

        // copy result back
        int h_recv;
        CUDA_CALL(cudaMemcpy(&h_recv, d_recv, sizeof(int), cudaMemcpyDeviceToHost));

        // rank 0 computes expected
        if (rank == 0) {
            int expected;
            switch (op) {
                case MPI_SUM:
                    expected = 0;
                    for (int r = 0; r < size; ++r) expected += (r+1);
                    break;
                case MPI_PROD:
                    expected = 1;
                    for (int r = 0; r < size; ++r) expected *= (r+1);
                    break;
                case MPI_MIN:
                    expected = INT_MAX;
                    for (int r = 0; r < size; ++r) if (r+1 < expected) expected = r+1;
                    break;
                case MPI_MAX:
                    expected = INT_MIN;
                    for (int r = 0; r < size; ++r) if (r+1 > expected) expected = r+1;
                    break;
                default:
                    expected = 0;
            }
            double avg_time_us = (t1 - t0) / NITER * 1e6;
            printf("%8s : result=%d, expected=%d, %8.2f µs/iter\n",
                   name, h_recv, expected, avg_time_us);
        }
    }

    CUDA_CALL(cudaFree(d_send));
    CUDA_CALL(cudaFree(d_recv));
    MPI_CALL(MPI_Finalize());
    return 0;
}

