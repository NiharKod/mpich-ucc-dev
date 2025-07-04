#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Print a hello world message from each process
  printf("Hello world from rank %d out of %d processors\n", world_rank, world_size);
  // Finalize the MPI environment
  MPI_Finalize();
  return 0;
}


