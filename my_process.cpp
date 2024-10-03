#include "mpi.h"
#include <fstream>
#include <chrono>

class Process {
private:
  std::chrono::steady_clock::time_point startTime, endTime;
  std::chrono::nanoseconds elapsedTime;
public:
  int totalProcesses = -1;
  int rank = -1;

  Process(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &this->totalProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
  }
  int getTotalProcesses() { return totalProcesses; }
  void setTotalProcesses(int totalProcesses) { this->totalProcesses = totalProcesses; }
  std::chrono::steady_clock::time_point getStartTime() { return this->startTime; }
  std::chrono::steady_clock::time_point getEndTime() { return this->endTime; }
  std::chrono::nanoseconds getElapsedTime() { return this->elapsedTime; }
  
  void startTimer() {
    this->startTime = std::chrono::steady_clock::now();
  }
  void endTimer() {
    this->endTime = std::chrono::steady_clock::now();
    this->elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(this->endTime - this->startTime);
  }
  
  virtual void run() = 0;
};

class SimpleProcess : public Process {
private: 
  bool isOutput;
  bool isRankAscending;

public:
  SimpleProcess(int argc, char *argv[], bool isOutput = false, bool isRankAscending = false) : 
    Process(argc, argv), 
    isOutput(isOutput),
    isRankAscending(isRankAscending) { 
    for (int i = 1; i < argc; i++) {
      std::string argument = argv[i];
      if (argument == "--output") this->isOutput = true;
      else if (argument == "--rank-ascending") this->isRankAscending = true;
    }
  }
  
  void output() {
    std::ofstream outfile;
    outfile.open("temp/output.txt", std::ios::app);
    outfile << "P=" << this->getTotalProcesses() << " | T=" << this->getElapsedTime().count() << " ns\n";
    outfile.close();
  }

  void receive() {
    int total = this->getTotalProcesses();
    int recvRank;
    MPI_Status recvStatus;
    printf("Total processes %d", total);
    printf("\nHello from process %3d", this->rank); // printing hello for process 0
    for (int i = 1; i < total; i++) // starting with 1 because this happends in the process 0
    {
      int dest = this->isRankAscending ? i : MPI_ANY_SOURCE;
      MPI_Recv(&recvRank, 1, MPI_INT, dest, MPI_ANY_TAG, MPI_COMM_WORLD, &recvStatus);
      printf("\nHello from process %3d", recvRank); // printing hello for received process
    }
  }

  void send() {
    MPI_Send(&this->rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); // send rank to process 0
  }

  void run() override {
    
    if (this->rank == 0) {
      this->startTimer();
      this->receive();
      this->endTimer();  
      if (this->isOutput) {
        this->output();
      }
    }
    else {
      send();
    }
    
    MPI_Finalize();
  }
};

class VectorProcess : public Process {
private:
  int *vectorA = nullptr;
  int *vectorB = nullptr;
  int vectorSize = -1;
public:
  VectorProcess(int argc, char *argv[]) : Process(argc, argv) { }
  ~VectorProcess() {
    delete[] vectorA;
    delete[] vectorB;
  }
  int getVectorSize() { return vectorSize; }
  int multiplyVectors() { 
     int result = 0;
    for (int i = 0; i < this->vectorSize; i++) {
      result += vectorA[i] * vectorB[i];
    }
    return result;   
  }

  /**
   * Input of two vectors
   */
  void init() {
    
  }

  /**
   * Main method of the algorithm
   */
  void run() override {
    init();


    // TODO: main method in algorithm
  }
};
