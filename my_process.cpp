#include "mpi.h"
#include <fstream>
#include <chrono>
#include <iostream>
#include <string>
#include "mpi.h"
#include <vector> 

class Logger { 
private:
  std::vector<std::string> logs;
protected:
  void write(std::string log) {
    this->logs.push_back(log);
  }
public:
  std::vector<std::string> getLogs() {
    return this->logs;
  }
};

class Process {
private:
  std::chrono::steady_clock::time_point startTime, endTime;
  std::chrono::nanoseconds elapsedTime;
protected:
  int totalProcesses = -1;
  int availableProcesses = -1;
  int rank = -1;
  void startTimer() {
    this->startTime = std::chrono::steady_clock::now();
  }
  void endTimer() {
    this->endTime = std::chrono::steady_clock::now();
    this->elapsedTime = std::chrono::duration_cast<std::chrono::nanoseconds>(this->endTime - this->startTime);
  }

public:
  Process(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &this->totalProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    if (this->rank == 0) this->availableProcesses = this->totalProcesses - 1;
  }
  int getTotalProcesses() { return totalProcesses; }
  std::chrono::steady_clock::time_point getStartTime() { return this->startTime; }
  std::chrono::steady_clock::time_point getEndTime() { return this->endTime; }
  std::chrono::nanoseconds getElapsedTime() { return this->elapsedTime; }
  
  
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
      // if we want to print elapsed time of whole process into the file
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
  std::vector<float> vectorA;
  std::vector<float> vectorB;
  int vectorSize = 0;
  float scalarSum = 0;

  std::vector<float> inputVector() {
    std::vector<float> new_vector;
    for (int i = 0; i < this->vectorSize; i++) {
      float new_value;
      std::cin >> new_value;
      new_vector.push_back(new_value);
    }
    return new_vector;
  }

public:
  VectorProcess(int argc, char *argv[]) : Process(argc, argv) { }
  ~VectorProcess() {
    // FIXME: maybe add here MPI_Finalize instead of run() method
  }
  
  std::vector<float> getVectorA() { return this->vectorA; }

  std::vector<float> getVectorB() { return this->vectorB; }

  int getVectorSize() { return vectorSize; }

  void init() {
    while(vectorSize == 0 || this->vectorA.empty() || this->vectorB.empty()) {
      try
      {
        std::cout << "Input vector size: ";
        std::cin >> this->vectorSize;

        std::cout << "Input vector A: \n";
        this->vectorA = this->inputVector();

        std::cout << "Input vector B: \n";
        this->vectorB = this->inputVector();
        break;
      }
      catch(const std::exception& e)
      {
        // FIXME: find why this block doesn't work when input is wrong !!!
        std::cerr << e.what() << '\n';
        this->vectorSize;
        this->vectorA.clear();
        this->vectorB.clear();
        continue;
      }
    }
  }

  float calculateScalar(std::vector<float> v1, std::vector<float> v2) {
    int size = v1.size();
    float result = 0;
    for (int i = 0; i < size; i++) {
      result += (v1[i] * v2[i]);
    }
    return result;
  }

  // receive for parent process
  // receives and returns calculated scalars from child processes
  float receive_scalar(int process_id) {
    float value;
    MPI_Status status;
    MPI_Recv(&value, 1, MPI_FLOAT, process_id, 0, MPI_COMM_WORLD, &status);
    return value;
  }

  // receive for CHILD process
  // receives and returns size for incoming vector
  int receive_v_size() {
    int size;
    MPI_Status status;
    MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    std::cout << "\nPROCESS " << this->rank << " RECEIVING VECTOR WITH SIZE " << size;
    return size;
  }
  // receive for CHILD process
  // receives and returns vector with values to calculate scalar value
  std::vector<float> receive_vec(int size) {
    std::vector<float> vec(size);
    MPI_Status status;
    MPI_Recv(vec.data(), size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
    return vec;
  }

  void send_vecs(int process_id, std::vector<float> vec1, std::vector<float> vec2) {
    std::cout 
        << "\nSENDING TO PROCESS #" 
        << process_id
        << "\n\tvector[0]: "
        << this->vectorToString(vec1)
        << "\n\tvector[1]: "
        << this->vectorToString(vec2)
        << "\n\tsize: " 
        << vec1.size()
        << std::endl;

    MPI_Send(vec1.data(), vec1.size(), MPI_FLOAT, process_id, 0, MPI_COMM_WORLD);
    MPI_Send(vec2.data(), vec2.size(), MPI_FLOAT, process_id, 0, MPI_COMM_WORLD);
  }

  void send_v_size(int process_id, int size) {
    MPI_Send(&size, 1, MPI_INT, process_id, 0, MPI_COMM_WORLD);
  }

  void send_scalar(float &sum) {
    MPI_Send(&sum, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }

  void run() override {
    
    if (this->totalProcesses == 1) {
      std:: cout << this->toString();
      init();
      std:: cout << this->toString();
      std:: cout << "\n\nscalar: " << this->calculateScalar(this->vectorA, this->vectorB);
    }
    else { 
      if (this->rank == 0) {
        init();
        std::cout << this->toString();
        int v_index = 0;
        int new_v_size = this->vectorSize < this->availableProcesses ? 1 : this->vectorSize / this->availableProcesses;
        std::vector<int> process_queue;

        for (int i = 0; i < this->availableProcesses; i ++ ) {
          int process_id = i + 1;
          if (this->vectorSize < process_id) {
            std:: cout << "\nSENDING SIZE -1 TO PROCESS " << process_id << std::endl;
            this->send_v_size(process_id, -1);
            continue;
          }
          std::vector<float> vec1;
          std::vector<float> vec2;

          for (int j = 0; j < new_v_size; j++) {
            vec1.push_back(vectorA[v_index]);
            vec2.push_back(vectorB[v_index]);
            v_index++;
          }

          std::cout << "\nSENDING SIZE " << vec1.size() << " TO PROCESS " << process_id;
          this->send_v_size(process_id, vec1.size());
          this->send_vecs(process_id, vec1, vec2);    
          process_queue.push_back(process_id);
        }

        int left_items = this->vectorSize - v_index; // v_index в данном случае по факту уже значит количество пересланных элементов
        
        if (left_items > 0) {
          std::vector<float> vec1;
          std::vector<float> vec2;

          for (int j = 0; j < left_items; j++) {
            vec1.push_back(vectorA[v_index]);
            vec2.push_back(vectorB[v_index]);
            v_index++;
          }

          this->scalarSum += this->calculateScalar(vec1, vec2);
        }
        std::cout << "ITEMS LEFT: " << left_items << " | CALCULATED SCALAR: " << this->scalarSum << std::endl;

        std::cout << "\nPROCESSES IN QUEUE" << this->vectorToString(process_queue) << std::endl; 

        std::cout << "\nPROCESS 0 RECEIVING SCALARS\n";
        for (int process_id : process_queue) {
          float received_scalar = this->receive_scalar(process_id);
          std::cout 
            << "!!! PROCESS 0 RECEIVED SCALAR " 
            << received_scalar 
            << " FROM PROCESS " 
            << process_id 
            << std::endl;
          this->scalarSum += received_scalar;
        }
        std::cout << "\n\n\n\n RESULT SCALAR SUM: " << this->scalarSum;
      }
      else {
        std::cout << this->toString();
        int size = this->receive_v_size();
        std::cout << "\nPROCESS " << this->rank << " RECEIVED SIZE " << size;
        
        if (size != -1) {
          std::vector<float> vec1 = this->receive_vec(size);
          std::vector<float> vec2 = this->receive_vec(size);
          std::cout 
            << "\nPROCESS " 
            << this->rank 
            << "\n\t RECEIVED VECTOR " 
            << this->vectorToString(vec1) 
            << "\n\t AND VECTOR " 
            << this->vectorToString(vec2);

          float scalar = this->calculateScalar(vec1, vec2);
          this->send_scalar(scalar);
        }
      }
    }
    

    MPI_Finalize();

  }

  template <typename T>
  std::string vectorToString(std::vector<T> vec) {
    std::string result = "";
    for (T item : vec) {
      result += std::to_string(item) + ", ";
    }
    return "[" + result + "]";
  }

  std::string toString() {
    // if object have vectors then convert them into strings
    if (!vectorA.empty() && !vectorB.empty()) {
      std::string *vectors = nullptr;
      vectors = new std::string[2]{"", ""};

      vectors[0] = this->vectorToString(this->vectorA);
      vectors[1] = this->vectorToString(this->vectorB);

      return  "\nHello, i'm process " + std::to_string(this->rank) + 
              "\nVectorA: " + vectors[0] + 
              "\nVectorB: " + vectors[1] + "\n";
    }

    // if object doesn't have any vectors    
    return  "\nHello, i'm process " + std::to_string(this->rank);
  }
};

 
