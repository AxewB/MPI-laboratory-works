#include "mpi.h"
#include <fstream>
#include <chrono>
#include <iostream>
#include <string>
#include "mpi.h"
#include <vector> 

#include <random>
#include <iomanip>  // Для std::setprecision и std::fixed

class Util {
public:
  static std::vector<float> generateRandomVector(int size, float lower_bound = 0, float upper_bound = 10) {
    std::mt19937 gen(static_cast<unsigned int>(std::time(0)));
    std::uniform_real_distribution<float> dist(lower_bound, upper_bound);

    std::vector<float> randomVector;
    randomVector.reserve(size);

    for (int i = 0; i < size; ++i) {
        randomVector.push_back(dist(gen));
    }

    return randomVector;
  }
};

// TODO: maybe remove this entirely
class Logger { 
private:
  static std::vector<std::string> logs;
protected:
  static void writeLog(std::string log, bool printToConsole) {
    logs.push_back(log);
    if (printToConsole) {
      std::cout << log;
    }
  }
public:
  static std::vector<std::string> getLogs() {
    return logs;
  }
  static std::string getStringLogs() {
    std::string result = "";
    for (std::string log : logs) 
      result += log + "\n";

    return result;
  }
};

class Process {
private:
  std::chrono::steady_clock::time_point startTime, endTime;
  std::chrono::nanoseconds elapsedTime;
  // FIXME: move output logic here
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

// mpiexec -n 6 main --random --size 10
class VectorProcess : public Process {
private:
  std::vector<float> vectorA;
  std::vector<float> vectorB;
  int vectorSize = 0;
  float scalarSum = 0;
  bool isRandom = false;

  bool isOutput = false;

  std::vector<float> inputVector() {
    std::vector<float> new_vector;
    for (int i = 0; i < this->vectorSize; i++) {
      float new_value;
      std::cin >> new_value;
      new_vector.push_back(new_value);
    }
    return new_vector;
  }

  void output() {
    if (!this->isOutput) 
      return;

    std::ofstream outfile;
    outfile.open("temp/lab-2-output.csv", std::ios::app);

    if (!outfile.is_open()) {
      std::cerr << "Error while opening file!" << std::endl;
      return;
    }
    // outfile 
    //   << "processes=" 
    //   << this->getTotalProcesses() 
    //   << " | elapsed_time=" 
    //   << this->getElapsedTime().count() 
    //   << " | vector_size=" 
    //   << this->vectorSize
    //   << "\n";
    // Проверяем, существует ли уже заголовок
    std::string line;
    bool headerExists = false;

    // Считываем первые строки файла, чтобы проверить наличие заголовка
    std::ifstream infile("temp/lab-2-output.csv");
    if (infile.good()) {
        std::getline(infile, line);
        if (line == "P,T,V") {
            headerExists = true;
        }
    }

    // Если заголовок отсутствует, записываем его
    if (!headerExists) {
        outfile << "P,T,V\n";
    }
    // Placing cursors in the end of the file
    outfile.seekp(0, std::ios::end);
    outfile 
      << this->getTotalProcesses() 
      << ", " 
      << this->getElapsedTime().count() 
      << ", " 
      << this->vectorSize
      << "\n";
    outfile.close();

    std::cout << "\nElapsed time: " << this->getElapsedTime().count();
  }
public:
  VectorProcess(int argc, char *argv[]) : Process(argc, argv) { 
    for (int i = 1; i < argc; i++) {
      std::string argument = argv[i];
      if (argument == "--size") {
        this->vectorSize = std::atoi(argv[i + 1]);
      }
      else if (argument == "--random") {
        this->isRandom = true;
      }
      else if (argument == "--output") {
        this->isOutput = true;
      }
    }
  }
  ~VectorProcess() {}
  
  std::vector<float> getVectorA() { return this->vectorA; }

  std::vector<float> getVectorB() { return this->vectorB; }

  int getVectorSize() { return vectorSize; }

  void init() {
    if (this->isRandom) {
      this->vectorA = Util::generateRandomVector(this->vectorSize);
      this->vectorB = Util::generateRandomVector(this->vectorSize);
    }
    else {
      while(this->vectorSize == 0 || this->vectorA.empty() || this->vectorB.empty()) {
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
    // std::cout << "\nPROCESS " << this->rank << " RECEIVING VECTOR WITH SIZE " << size;
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
    // std::cout 
    //     << "\nSENDING TO PROCESS #" 
    //     << process_id
    //     << "\n\tvector[0]: "
    //     << this->vectorToString(vec1)
    //     << "\n\tvector[1]: "
    //     << this->vectorToString(vec2)
    //     << "\n\tsize: " 
    //     << vec1.size()
    //     << std::endl;

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
      this->startTimer();
      std:: cout << this->toString();
      init();
      this->scalarSum = this->calculateScalar(this->vectorA, this->vectorB);
      std:: cout << "\n\nCALCULATED SCALAR: " << this->scalarSum;

      this->endTimer();
      this->output();
    }
    else { 
      if (this->rank == 0) {
        init();
        std::cout << this->toString();
        int v_index = 0;
        int new_v_size = this->vectorSize < this->availableProcesses ? 1 : this->vectorSize / this->availableProcesses;
        std::vector<int> process_queue;

        // Starting timer after initialization
        this->startTimer();

        for (int i = 0; i < this->availableProcesses; i ++ ) {
          int process_id = i + 1;
          if (this->vectorSize < process_id) {
            // std:: cout << "\nSENDING SIZE -1 TO PROCESS " << process_id << std::endl;
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

          // std::cout << "\nSENDING SIZE " << vec1.size() << " TO PROCESS " << process_id;
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
        // std::cout << "ITEMS LEFT: " << left_items << " | CALCULATED SCALAR: " << this->scalarSum << std::endl;

        // std::cout << "\nPROCESSES IN QUEUE" << this->vectorToString(process_queue) << std::endl; 

        // std::cout << "\nPROCESS 0 RECEIVING SCALARS\n";
        for (int process_id : process_queue) {
          float received_scalar = this->receive_scalar(process_id);
          // std::cout 
          //   << "!!! PROCESS 0 RECEIVED SCALAR " 
          //   << received_scalar 
          //   << " FROM PROCESS " 
          //   << process_id 
          //   << std::endl;
          this->scalarSum += received_scalar;
        }
        std::cout << "\n\n\n\n RESULT SCALAR SUM: " << this->scalarSum;
        this->endTimer();
        this->output();
      }
      else {
        std::cout << this->toString();
        int size = this->receive_v_size();
        // std::cout << "\nPROCESS " << this->rank << " RECEIVED SIZE " << size;
        
        if (size != -1) {
          std::vector<float> vec1 = this->receive_vec(size);
          std::vector<float> vec2 = this->receive_vec(size);
          // std::cout 
          //   << "\nPROCESS " 
          //   << this->rank 
          //   << "\n\t RECEIVED VECTOR " 
          //   << this->vectorToString(vec1) 
          //   << "\n\t AND VECTOR " 
          //   << this->vectorToString(vec2);

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

  std::string toString(bool printVector = false) {
    // if object have vectors then convert them into strings
    if (!vectorA.empty() && !vectorB.empty() && printVector) {
      std::string *vectors = nullptr;
      vectors = new std::string[2]{"", ""};

      vectors[0] = this->vectorToString(this->vectorA);
      vectors[1] = this->vectorToString(this->vectorB);

      return  "\nHello, i'm process " + std::to_string(this->rank) + 
              "\n\tVectorA: " + vectors[0] + 
              "\n\tVectorB: " + vectors[1] + "\n";
    }

    // if object doesn't have any vectors    
    return  "\nHello, i'm process " + std::to_string(this->rank);
  }
};

// Можно сделать класс в классе (родительский и дочерний в классе NetworkProcess), у которых будут определены методы send и receive

class NetworkProcess : public Process {
private: 
  struct Packet {
    int destination;
    int data;
    MPI_Status status;
  };
  // Маршрутизатор и клиент
  class Router {

  };
  class Client {

  };

  int getRandomDestination() {
    return rand() % totalProcesses;
  }
public:
  NetworkProcess(int argc, char *argv[]) : Process(argc, argv) {}
  void run() {
    srand(time(0) * rank); // initialize random seed

    if (this->rank == 0) {
      Packet packet;
      std::vector<Packet> packets;

      // Receiving packets from other processes
      for (int i = 1; i < this->totalProcesses; i++){
        MPI_Recv(&packet, sizeof(packet), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &packet.status);
        packets.push_back(packet);
        
        std::cout 
          << "Process 0 received packet from " << packet.status.MPI_SOURCE 
          << " with destination " << packet.destination 
          << " and data: " << packet.data 
          << std::endl;
      }

      // Sending packet to destination
      for (Packet p : packets) {
        MPI_Send(&p, sizeof(p), MPI_BYTE, p.destination, 0, MPI_COMM_WORLD);

        // Receive confirmation report
        int confirmation;
        MPI_Recv(&confirmation, 1, MPI_INT, p.destination, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::cout 
          << "Process 0 confirmed receiving by process " << p.destination 
          << " (source ) " << p.status.MPI_SOURCE << ")"
          << std::endl;
        
        // Send confirmation report 
        confirmation = 1;
        std::cout 
          << "Process 0 sent confirmation to process " << p.destination 
          << " (from process " << p.status.MPI_SOURCE << ")"
          << std::endl;
        MPI_Send(&confirmation, 1, MPI_INT, p.destination, 0, MPI_COMM_WORLD);

      }

      // Ending all processes by passing packet with destination == -1
      for (int i = 1; i < this->totalProcesses; i++) {
        packet.destination = -1;
        std::cout << "Ending process " << i << std::endl;
        MPI_Send(&packet, sizeof(packet), MPI_BYTE, i, 0, MPI_COMM_WORLD);
      }
    }
    else {
      // Generating values
      Packet packet;
      packet.data = rand() % 1000;
      
      // Generating values while it's not process 0 or not the same as current process rank
      do {
        packet.destination = rand() % this->totalProcesses;
      } while (packet.destination == 0 || packet.destination == this->rank);
      MPI_Send(&packet, sizeof(packet), MPI_BYTE, 0, 0, MPI_COMM_WORLD);

      std::cout 
        << "Process " << this->rank 
        << " sent packet to 0"
        << " with destination " << packet.destination 
        << " and data: " << packet.data 
        << std::endl;

      while (true) {
        // Receiving packet from process 0
        MPI_Recv(&packet, sizeof(packet), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // If received destionation == -1 then leave loop and end process
        if (packet.destination == -1)
          break;

        std::cout 
          << "Process " << this->rank << " received packet from 0"
          << " with destination " << packet.destination 
          << " (source: " << packet.status.MPI_SOURCE << ")"
          << " and data: " << packet.data 
          << std::endl;

        // Sending confirmation to process 0 (int value = 1)
        {
          int confirmation = 1;
          MPI_Send(&confirmation, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
          std::cout << "Process " << this->rank << " sent confirmation to 0" << std::endl;
        }
        // Receiving confirmation from process 1 
        // int confirmation;
        // MPI_Recv(&confirmation, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // std::cout << "Process " << this->rank << " received confirmation from 0" << std::endl;  
      }

      
    }
    
    MPI_Finalize();
  }

};
 
