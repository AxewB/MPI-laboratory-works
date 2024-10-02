#include <mpi.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>

bool output_elapsed_time = false;

int main(int argc, char *argv[]) {
    int procRank, size;
    
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank); // getting rank of current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // getting total number of processes


    if (procRank == 0) {
      // Setting starting time
      auto start_time = std::chrono::steady_clock::now(); 
      int recvRank;
      MPI_Status recvStatus;
      printf("Total processes %d", size);
      printf("\nHello from process %3d", procRank); // printing hello for process 0
      for (int i = 1; i < size; i++) // starting with 1 because this happends in the process 0
      {
        MPI_Recv(&recvRank, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recvStatus);
        printf("\nHello from process %3d", recvRank); // printing hello for received process
      }

      // Getting end time and calculating execution time
      auto end_time = std::chrono::steady_clock::now();
      auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
      std::cout << "\nTime taken: " << elapsed_ns.count() << " ns\n";

      // Output elapsed time to a file if specified in the command line arguments
      if (output_elapsed_time) {
        std::ofstream outfile;
        outfile.open("temp/output.txt", std::ios::app);
        outfile << "P=" << size << " | T=" << elapsed_ns.count() << " ns\n";
        outfile.close();
      }
    }
    else {
      MPI_Send(&procRank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); // send procRank to process 0
    }

    MPI_Finalize();
    return 0;
}