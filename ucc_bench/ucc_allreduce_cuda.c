#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <ucc/api/ucc.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#define UCC_CHECK(status)                          \
    do {                                           \
        ucc_status_t ucc_status = status;          \
        if ((ucc_status != UCC_OK)) {              \
            fprintf(stderr, "UCC error at %s:%d\n", __FILE__, __LINE__); \
            exit(1);                               \
        }                                          \
    } while (0)

typedef struct {
    MPI_Comm comm;
    int rank;
} oob_ctx_t;

ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                           void *coll_ctx, void **req)
{
    oob_ctx_t *ctx = (oob_ctx_t *)coll_ctx;
    MPI_Request *mpi_req = malloc(sizeof(MPI_Request));
    MPI_Iallgather(sbuf, msglen, MPI_BYTE,
                   rbuf, msglen, MPI_BYTE,
                   ctx->comm, mpi_req);
    *req = mpi_req;
    return UCC_OK;
}

ucc_status_t oob_test(void *req)
{
    int completed;
    MPI_Test((MPI_Request *)req, &completed, MPI_STATUS_IGNORE);
    return completed ? UCC_OK : UCC_INPROGRESS;
}

ucc_status_t oob_free(void *req)
{
    free(req);
    return UCC_OK;
}

uint64_t correct_answer(int n_ranks, int bytes) {
    uint64_t element_wise_sum = (n_ranks * (n_ranks - 1)) / 2;
    return element_wise_sum * bytes;
}

int main(int argc, char **argv)
{
    int buff_size = 0;
    if (argc == 3 && strcmp(argv[1], "-n") == 0) {
        buff_size = atoi(argv[2]);
    } else {
        fprintf(stderr, "Usage: %s -n [buffer size]\n", argv[0]);
        exit(1);
    }

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* CUDA CHECKS */


    int device_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&device_count);  // Already declared here
    if (cerr != cudaSuccess || device_count == 0) {
        fprintf(stderr, "Rank %d: No CUDA-capable device found or error: %s\n",
                rank, cudaGetErrorString(cerr));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int device_id = rank % device_count;
    cerr = cudaSetDevice(device_id);  // reuse existing cerr
    if (cerr != cudaSuccess) {
        fprintf(stderr, "Rank %d: Failed to set CUDA device %d: %s\n",
                rank, device_id, cudaGetErrorString(cerr));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    struct cudaDeviceProp prop;
    cerr = cudaGetDeviceProperties(&prop, device_id);  // still reuse cerr

    if (cerr == cudaSuccess) {
        printf("Rank %d using CUDA device %d: %s (Compute Capability %d.%d)\n",
            rank, device_id, prop.name, prop.major, prop.minor);
        fflush(stdout);  // ensure it prints in MPI programs
    } else {
        fprintf(stderr, "Rank %d: Failed to get device properties: %s\n",
                rank, cudaGetErrorString(cerr));
    }


    // UCC library setup
    ucc_lib_h lib;
    ucc_lib_config_h lib_config;
    UCC_CHECK(ucc_lib_config_read(NULL, NULL, &lib_config));

    ucc_lib_params_t lib_params = {
        .mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCC_THREAD_SINGLE
    };
    UCC_CHECK(ucc_init(&lib_params, lib_config, &lib));
    ucc_lib_config_release(lib_config);

    // UCC context with OOB
    ucc_context_config_h ctx_config;
    ucc_context_h ctx;
    UCC_CHECK(ucc_context_config_read(lib, NULL, &ctx_config));

    oob_ctx_t oob_ctx = {MPI_COMM_WORLD, rank};
    ucc_context_oob_coll_t oob = {
        .allgather = oob_allgather,
        .req_test  = oob_test,
        .req_free  = oob_free,
        .coll_info = &oob_ctx,
        .n_oob_eps = size,
        .oob_ep    = rank
    };

    ucc_context_params_t ctx_params = {
        .mask = UCC_CONTEXT_PARAM_FIELD_TYPE | UCC_CONTEXT_PARAM_FIELD_OOB,
        .type = UCC_CONTEXT_EXCLUSIVE,
        .oob  = oob
    };
    UCC_CHECK(ucc_context_create(lib, &ctx_params, ctx_config, &ctx));
    ucc_context_config_release(ctx_config);

    // Create team
    ucc_team_params_t team_params = {
        .mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE | UCC_TEAM_PARAM_FIELD_OOB,
        .ep = rank,
        .ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG,
        .oob = oob
    };

    ucc_team_h team;
    ucc_context_h contexts[] = {ctx};
    UCC_CHECK(ucc_team_create_post(contexts, 1, &team_params, &team));
    while (ucc_team_create_test(team) == UCC_INPROGRESS) {
        UCC_CHECK(ucc_context_progress(ctx));
    }

    // Allocate device memory using CUDA
    uint8_t *src_buff, *dst_buff;
//    cudaError_t cerr;
    cerr = cudaMalloc((void **)&src_buff, buff_size);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc src_buff failed: %s\n", cudaGetErrorString(cerr));
        exit(1);
    }
    cerr = cudaMalloc((void **)&dst_buff, buff_size);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dst_buff failed: %s\n", cudaGetErrorString(cerr));
        exit(1);
    }

    cudaMemset(src_buff, rank, buff_size);

    // Setup UCC Allreduce collective
    ucc_coll_args_t coll_args = {
        .mask = 0,
        .coll_type = UCC_COLL_TYPE_ALLREDUCE,
        .src = {
            .info = {
                .buffer = src_buff,
                .count = buff_size,
                .datatype = UCC_DT_INT8,
                .mem_type = UCC_MEMORY_TYPE_CUDA
            }
        },
        .dst = {
            .info = {
                .buffer = dst_buff,
                .count = buff_size,
                .datatype = UCC_DT_INT8,
                .mem_type = UCC_MEMORY_TYPE_CUDA
            }
        },
        .op = UCC_OP_SUM,
        .flags = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                 UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER
    };

    ucc_coll_req_h req;
    UCC_CHECK(ucc_collective_init(&coll_args, &req, team));
    UCC_CHECK(ucc_collective_post(req));

    while (ucc_collective_test(req) == UCC_INPROGRESS) {
        UCC_CHECK(ucc_context_progress(ctx));
    }
    UCC_CHECK(ucc_collective_finalize(req));

    // Copy result to host and verify
    uint8_t *host_result = (uint8_t *)malloc(buff_size);
    cudaMemcpy(host_result, dst_buff, buff_size, cudaMemcpyDeviceToHost);

    uint64_t sum = 0;
    for (size_t i = 0; i < buff_size; i++) {
        sum += host_result[i];
    }
    printf("Rank %d: Allreduce result = %lu\n", rank, sum);

    if (rank == 0) {
        printf("Correct sum = %lu\n", correct_answer(size, buff_size));
    }

    // Cleanup
    free(host_result);
    cudaFree(src_buff);
    cudaFree(dst_buff);
    UCC_CHECK(ucc_team_destroy(team));
    UCC_CHECK(ucc_context_destroy(ctx));
    UCC_CHECK(ucc_finalize(lib));

    MPI_Finalize();
    return 0;
}

