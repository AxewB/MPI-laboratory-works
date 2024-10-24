#include <fstream>
#include <iomanip> // Для std::setprecision и std::fixed
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "mpi.h"

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

class Process {
private:
  double startTime, endTime, elapsedTime;
  bool isOutput = false;

protected:
  int totalProcesses = -1;
  int availableProcesses = -1;
  int rank = -1;
  void startTimer() {
    this->startTime = MPI_Wtime();
  }
  void endTimer() {
    this->endTime = MPI_Wtime();
    this->elapsedTime = this->endTime - this->startTime;
  }
  void output(std::string fileName = "output.csv",
              std::map<std::string, std::string> additionalValues = std::map<std::string, std::string>()) {
    if (!this->isOutput)
      return;

    std::string outputPath = "output/" + fileName;
    std::ofstream outfile;
    std::string header = "P,T";
    std::string data = std::to_string(this->totalProcesses) + "," + std::to_string(this->elapsedTime);

    // Opening file
    outfile.open(outputPath, std::ios::app);
    if (!outfile.is_open()) {
      std::cerr << "Error while opening file! Check if 'output' directory "
                   "exists and file is not opened!"
                << std::endl;
      return;
    }

    // Adding more info to header and data if exists
    for (const auto &pair : additionalValues) {
      header += "," + pair.first;
      data += "," + pair.second;
    }

    // Check if header exists
    std::string line;
    bool headerExists = false;

    // Reading first lines of the file to check header existance
    std::ifstream infile(outputPath);
    if (infile.good()) {
      std::getline(infile, line);
      if (line == header) {
        headerExists = true;
      }
    }

    // If there no header then add it
    if (!headerExists)
      outfile << header << "\n";

    // Placing cursor in the end of the file and printing data into it
    outfile.seekp(0, std::ios::end);
    outfile << data << "\n";
    outfile.close();

    std::cout << "\nElapsed time: " << this->getElapsedTime();
  }

public:
  Process(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &this->totalProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    if (this->rank == 0)
      this->availableProcesses = this->totalProcesses - 1;

    for (int i = 0; i < argc; i++) {
      std::string argument = argv[i];
      if (argument == "--output")
        this->isOutput = true;
    }
  }
  int getTotalProcesses() { return totalProcesses; }
  double getStartTime() { return this->startTime; }
  double getEndTime() { return this->endTime; }
  double getElapsedTime() { return this->elapsedTime; }

  virtual void run() = 0;
};

// LR1
class SimpleProcess : public Process {
private:
  bool isRankAscending = false;
  void printHello(int process) { std::cout << "Hello from process " << process << std::endl; }

public:
  SimpleProcess(int argc, char *argv[], bool isOutput = false, bool isRankAscending = false) : Process(argc, argv) {
    for (int i = 1; i < argc; i++) {
      std::string argument = argv[i];
      if (argument == "--rank-ascending")
        this->isRankAscending = true;
    }
  }

  void receive() {
    int recvRank;
    MPI_Status recvStatus;
    std::cout << "Total processes " << this->totalProcesses << std::endl;

    // printing hello for process 0
    this->printHello(this->rank);

    // starting with 1 because this happends in the process 0
    for (int i = 1; i < this->totalProcesses; i++) {
      int dest = this->isRankAscending ? i : MPI_ANY_SOURCE;
      MPI_Recv(&recvRank, 1, MPI_INT, dest, MPI_ANY_TAG, MPI_COMM_WORLD, &recvStatus);

      this->printHello(recvRank);
    }
  }

  // send rank to process 0
  void send() { MPI_Send(&this->rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD); }

  void run() override {
    if (this->rank == 0) {
      this->startTimer();
      this->receive();
      this->endTimer();

      // print elapsed time into file if passed '--output' argument in terminal
      this->output("lab-1.output.csv");
    } else {
      send();
    }

    MPI_Finalize();
  }
};

// LR2
// mpiexec -n 6 main --random --size 10
class VectorProcess : public Process {
private:
  std::vector<float> vectorA;
  std::vector<float> vectorB;
  int vectorSize = 0;
  float scalarSum = 0;
  bool isRandom = false;

  // bool isOutput = false;

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
  VectorProcess(int argc, char *argv[]) : Process(argc, argv) {
    for (int i = 1; i < argc; i++) {
      std::string argument = argv[i];
      if (argument == "--size") {
        this->vectorSize = std::atoi(argv[i + 1]);
      } else if (argument == "--random") {
        this->isRandom = true;
      }

      if (this->isRandom && this->vectorSize == 0) {
        throw std::runtime_error("\nERROR: There is no size declared but random is passed. Please "
                                 "consider passing '--size value' when using '--random'");
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
    } else {
      while (this->vectorSize == 0 || this->vectorA.empty() || this->vectorB.empty()) {
        try {
          std::cout << "Input vector size: ";
          std::cin >> this->vectorSize;

          std::cout << "Input vector A: \n";
          this->vectorA = this->inputVector();

          std::cout << "Input vector B: \n";
          this->vectorB = this->inputVector();
          break;
        } catch (const std::exception &e) {
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
    // std::cout << "\nPROCESS " << this->rank << " RECEIVING VECTOR WITH SIZE "
    // << size;
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

  void send_v_size(int process_id, int size) { MPI_Send(&size, 1, MPI_INT, process_id, 0, MPI_COMM_WORLD); }

  void send_scalar(float &sum) { MPI_Send(&sum, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD); }

  void run() override {
    if (this->totalProcesses == 1) {
      this->startTimer();
      std::cout << this->toString();
      init();
      this->scalarSum = this->calculateScalar(this->vectorA, this->vectorB);
      std::cout << "\n\nCALCULATED SCALAR: " << this->scalarSum;

      this->endTimer();
      this->output("lab-2.output.csv", std::map<std::string, std::string>{{"V", std::to_string(this->vectorSize)}});
    } else {
      if (this->rank == 0) {
        init();
        std::cout << this->toString();
        int v_index = 0;
        int new_v_size = this->vectorSize < this->availableProcesses ? 1 : this->vectorSize / this->availableProcesses;
        std::vector<int> process_queue;

        // Starting timer after initialization
        this->startTimer();

        for (int i = 0; i < this->availableProcesses; i++) {
          int process_id = i + 1;
          if (this->vectorSize < process_id) {
            // std:: cout << "\nSENDING SIZE -1 TO PROCESS " << process_id <<
            // std::endl;
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

          // std::cout << "\nSENDING SIZE " << vec1.size() << " TO PROCESS " <<
          // process_id;
          this->send_v_size(process_id, vec1.size());
          this->send_vecs(process_id, vec1, vec2);
          process_queue.push_back(process_id);
        }

        int left_items = this->vectorSize - v_index; // v_index в данном случае по факту уже
                                                     // значит количество пересланных элементов

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
        // std::cout << "ITEMS LEFT: " << left_items << " | CALCULATED SCALAR: "
        // << this->scalarSum << std::endl;

        // std::cout << "\nPROCESSES IN QUEUE" <<
        // this->vectorToString(process_queue) << std::endl;

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
        std::cout << "vector size" << this->vectorSize << std::endl;
        this->output("lab-2.output.csv", std::map<std::string, std::string>{{"V", std::to_string(this->vectorSize)}});
      } else {
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

  template <typename T> std::string vectorToString(std::vector<T> vec) {
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

      return "\nHello, i'm process " + std::to_string(this->rank) + "\n\tVectorA: " + vectors[0] +
             "\n\tVectorB: " + vectors[1] + "\n";
    }

    // if object doesn't have any vectors
    return "\nHello, i'm process " + std::to_string(this->rank);
  }
};

// LR3
// mpiexec -n 6 main
class NetworkProcess : public Process {
private:
  // Packet struct to send messages
  struct Packet {
    int destination;
    int data;
    MPI_Status status;
  };
  // Router is process with rank 0
  struct Router {
    void receivePackets(NetworkProcess *np, std::vector<Packet> &packets) {
      Packet packet;

      // Receiving packets from other processes
      for (int i = 1; i < np->totalProcesses; i++) {
        MPI_Recv(&packet, sizeof(Packet), MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &packet.status);
        packets.push_back(packet);

        std::cout << "Process 0 received packet from " << packet.status.MPI_SOURCE << " with destination "
                  << packet.destination << " and data: " << packet.data << std::endl;
      }
    }
    void sendPacket(NetworkProcess *np, Packet &packet) {
      MPI_Send(&packet, sizeof(packet), MPI_BYTE, packet.destination, 0, MPI_COMM_WORLD);
    }
    void receiveConfirmation(NetworkProcess *np, Packet &packet) {
      int confirmation;
      MPI_Recv(&confirmation, 1, MPI_INT, packet.destination, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::cout << "Process 0 confirmed receiving by process " << packet.destination << " (source "
                << packet.status.MPI_SOURCE << ")" << std::endl;
    }

    // Send confirmation report
    void sendConfirmation(NetworkProcess *np, Packet &packet) {
      int confirmation = 1;
      std::cout << "Process 0 sent confirmation to process " << packet.destination << " (from process "
                << packet.status.MPI_SOURCE << ")" << std::endl;
      MPI_Send(&confirmation, 1, MPI_INT, packet.destination, 0, MPI_COMM_WORLD);
    }
    void endAllProcesses(NetworkProcess *np) {
      Packet packet;
      for (int i = 1; i < np->totalProcesses; i++) {
        packet.destination = -1;
        std::cout << "Ending process " << i << std::endl;
        MPI_Send(&packet, sizeof(Packet), MPI_BYTE, i, 0, MPI_COMM_WORLD);
      }
    }
  };
  // Client is processes with rank > 0
  struct Client {
    void generateData(NetworkProcess *np, Packet &packet) {
      packet.data = rand() % 1000;

      // Generating values while it's not process 0 or not the same as current process rank
      do {
        packet.destination = rand() % np->totalProcesses;
      } while (packet.destination == 0 || packet.destination == np->rank);
    }
    void sendPacket(NetworkProcess *np, Packet &packet) {
      MPI_Send(&packet, sizeof(Packet), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
      std::cout << "Process " << np->rank << " sent packet to 0"
                << " with destination " << packet.destination << " and data: " << packet.data << std::endl;
    }
    // receving packet with destination -1 means exit for process
    int receivePacket(NetworkProcess *np, Packet &packet) {
      MPI_Recv(&packet, sizeof(Packet), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::cout << "Process " << np->rank << " received packet from 0"
                << " with destination " << packet.destination << " (source: " << packet.status.MPI_SOURCE << ")"
                << " and data: " << packet.data << std::endl;
      return packet.destination;
    }
    void sendConfirmation(NetworkProcess *np) {
      int confirmation = 1;
      MPI_Send(&confirmation, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      std::cout << "Process " << np->rank << " sent confirmation to 0" << std::endl;
    }
    void receiveConfirmation(NetworkProcess *np) {
      int confirmation;
      MPI_Recv(&confirmation, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::cout << "Process " << np->rank << " received confirmation from 0" << std::endl;
    }
  };

  int getRandomDestination() { return rand() % totalProcesses; }

public:
  NetworkProcess(int argc, char *argv[]) : Process(argc, argv) {}
  void run() {
    srand(time(0) * rank); // initialize random seed

    if (this->rank == 0) {
      this->startTimer();
      std::vector<Packet> packets;
      Router router;
      // Receiving packets from other processes
      router.receivePackets(this, packets);

      // Sending packet to destination
      for (Packet packet : packets) {
        router.sendPacket(this, packet);
        router.receiveConfirmation(this, packet);
        router.sendConfirmation(this, packet);
      }

      // Ending all processes by passing packet with destination == -1
      router.endAllProcesses(this);
      this->endTimer();
      this->output("lab-3.output.csv");
    } else {
      Packet packet;
      Client client;

      // Generating values
      client.generateData(this, packet);
      client.sendPacket(this, packet);
      while (true) {
        // Receiving packet from process 0
        // If received destination == -1 then leave loop and end process
        if (client.receivePacket(this, packet) == -1)
          break;

        client.sendConfirmation(this);
        client.receiveConfirmation(this);
      }
    }

    MPI_Finalize();
  }
};

// LR4
// TODO: maybe change vector<int> to vector<Packet> in which each packet will
// contain (rank, value) where rank is sourceRank and value is generated number
class CollectiveProcess : public Process {
private:
  void generateValues(std::vector<int> &values) {
    srand(time(0) * this->rank);
    for (int i = 0; i < this->totalProcesses; i++) {
      int value = rand();
      values.push_back(value);
    }
  }
  void generateValuesFromRank(std::vector<int> &values) {
    srand(time(0) * this->rank);
    for (int i = 0; i < this->totalProcesses; i++) {
      values.push_back(this->rank * std::pow(10, i));
    }
  }
  void printValues(std::vector<int> &values) {
    for (int number : values) {
      std::cout << number << " ";
    }
    std::cout << std::endl;
  }

public:
  CollectiveProcess(int argc, char *argv[]) : Process(argc, argv) {}
  void run() {
    std::vector<int> sendBuffer, recvBuffer(this->totalProcesses);
    // generating values to send
    this->generateValues(sendBuffer);

    // starting timer after numbers initialization
    this->startTimer();

    std::cout << "Process " << rank << " sending numbers: ";
    this->printValues(sendBuffer);

    // sending data
    MPI_Alltoall(sendBuffer.data(), 1, MPI_INT, recvBuffer.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // Выводим результаты
    std::cout << "Process " << rank << " received numbers: ";
    this->printValues(recvBuffer);

    this->endTimer();

    // Output one by one, otherwise this will cause errors while pushing strings into one file
    if (this->rank == 0) {
      this->output();
      int confirmationOutput = 1;
      MPI_Send(&confirmationOutput, 1, MPI_INT, this->rank + 1, 0, MPI_COMM_WORLD);
    } else {
      int confirmationOutput = 0;
      MPI_Recv(&confirmationOutput, 1, MPI_INT, this->rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (confirmationOutput == 1) {
        this->output("lab-4.output.csv");
      }
      if (this->rank != this->totalProcesses - 1) {
        MPI_Send(&confirmationOutput, 1, MPI_INT, this->rank + 1, 0, MPI_COMM_WORLD);
      }
    }
    MPI_Finalize();
  };
};

// LR5
class GroupProcess : public Process {
private:
  bool N;
  double A;

protected:
  int groupRank;
  MPI_Comm comm;

public:
  GroupProcess(int argc, char *argv[]) : Process(argc, argv) {
    srand(time(0) * this->rank);

    bool isNIsPresented = false;
    // min and max value for later generating A value if needed
    int min = 0;
    int max = 10;

    for (int i = 0; i < argc; i++) {
      std::string argument = argv[i];
      if (argument == "-N") {
        this->N = std::atoi(argv[i + 1]);
        isNIsPresented = true;
      } else if (argument == "--min")
        min = std::atoi(argv[i + 1]);
      else if (argument == "--max")
        max = std::atoi(argv[i + 1]);
    }

    if (!isNIsPresented)
      throw std::runtime_error("ERROR: N was not presented. Pass -N 'value' in the terminal");
    if (min >= max)
      throw std::runtime_error("ERROR: min equal or more than max value");

    // randomizing N value if it wasn't presented previously
    if (!this->N)
      this->N = rand() % 2;

    // if N is true then generate value for A to sum it later in whole group
    if (this->N)
      this->A = min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  }
  void run() {
    this->startTimer();
    // splitting process into different groups
    int color = this->N ? 1 : MPI_UNDEFINED;
    MPI_Comm_split(MPI_COMM_WORLD, color, this->rank, &this->comm);

    if (this->N) {
      // getting new rank of the process
      MPI_Comm_rank(this->comm, &this->groupRank);
      // waiting for other processes
      MPI_Barrier(this->comm);

      // calling MPI_Allreduce to count sum of A's for processes with N == true
      double sum = 0;
      MPI_Allreduce(&this->A, &sum, 1, MPI_DOUBLE, MPI_SUM, this->comm);
      std::cout << "Process " << this->rank << " (gRank: " << this->groupRank << ")"
                << " result sum: " << sum << " [N: " << this->N << ", "
                << "A: " << this->A << "]" << std::endl;
    } else
      std::cout << "Process " << this->rank << " (gRank: " << this->groupRank << ") wasn't summing";
    this->endTimer();
    this->output("lab-5.output.csv");
    MPI_Finalize();
  }
};

// LR6
class TopologyProcess : public Process {
public:
  TopologyProcess(int argc, char *argv[]) : Process(argc, argv) {}
  void run() { throw std::runtime_error("This class is not implemented yet"); }
};

// LR7
class MatrixProcess : public Process {
public:
  MatrixProcess(int argc, char *argv[]) : Process(argc, argv) {}

  void run() { throw std::runtime_error("This class is not implemented yet"); }
};