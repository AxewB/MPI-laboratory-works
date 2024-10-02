#include <mpi.h>
#include <stdio.h>


int main(int argc, char *argv[]) {
    int procRank, size;

    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank); // getting rank of current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // getting total number of processes


    if (procRank == 0) {
      int received_threads = 1; // 1 thread is proccess 0
      int recvRank;
      MPI_Status recvStatus;
      printf("Total processes %d", size);
      printf("\nHello from process %3d", procRank); // printing hello for process 0
      for (int i = 1; i < size; i++)
      {
        MPI_Recv(&recvRank, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recvStatus);
        printf("\nHello from process %3d", recvRank); // printing hello for received process
      }
    }
    else {
      MPI_Send(&procRank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); // send procRank to process 0
    }

    MPI_Finalize();
    return 0;
}