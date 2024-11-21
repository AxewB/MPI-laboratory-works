#include "mpi.h"
#include <fstream>
#include <iomanip> // Для std::setprecision и std::fixed
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>
#include <sstream>

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
  static double generateDoubleValue(double min = 0, double max = 100) {
    return min + static_cast<double>(rand()) / RAND_MAX * (max - min);
  }

  static std::vector<std::vector<double>> generateRandomMatrix(int size) {
    std::vector<std::vector<double>> resultMatrix(size, std::vector<double>(size));
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        resultMatrix[i][j] = rand() % 10;
      }
    }
    return resultMatrix;
  }
  static std::vector<std::vector<double>> generateLinearMatrix(int size, bool reverse = false) {
    srand(time(0) * 3); // setting random seed
    std::vector<std::vector<double>> resultMatrix(size, std::vector<double>(size));
    double newValue = reverse ? size * size : 1;
    for (int i = 0; i < size; i++)
      for (int j = 0; j < size; j++)
        resultMatrix[i][j] = reverse ? newValue-- : newValue++;

    return resultMatrix;
  }
  static void printMatrix(std::vector<std::vector<int>> &matrix) {
    for (const auto &row : matrix) {
      for (int value : row) {
        std::cout << value << " ";
      }
      std::cout << "\n";
    }
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
  void startTimer() { this->startTime = MPI_Wtime(); }
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
  }
  void printElapsedTime() {
    std::cout << "\n[Process " << this->rank << "] " << "Elapsed time " << this->getElapsedTime();
  }
  double calculateMeanElapsedTime(MPI_Comm comm) {
    // calculating mean time (actual mean will count process 0 later)
    int size = 0;
    double elapsed_time = this->getElapsedTime();
    double mean_time;

    MPI_Comm_size(comm, &size);
    // reducing time to process 0 of the group
    MPI_Reduce(&elapsed_time, &mean_time, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    return mean_time / size;
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
      this->printElapsedTime();
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
        std::cout << "\n[Process " << this->rank << "] " << "Total scalar: " << this->scalarSum;
        this->endTimer();
        // std::cout << "vector size" << this->vectorSize << std::endl;
        this->output("lab-2.output.csv", std::map<std::string, std::string>{{"V", std::to_string(this->vectorSize)}});
        this->printElapsedTime();
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
      this->printElapsedTime();
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
      this->output("lab-4.output.csv");
      this->printElapsedTime();
      int confirmationOutput = 1;
      MPI_Send(&confirmationOutput, 1, MPI_INT, this->rank + 1, 0, MPI_COMM_WORLD);
    } else {
      int confirmationOutput = 0;
      MPI_Recv(&confirmationOutput, 1, MPI_INT, this->rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (confirmationOutput == 1) {
        this->output("lab-4.output.csv");
        this->printElapsedTime();
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
  int group;
  int groupRank;
  int groupSize;
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
        if (this->rank == std::atoi(argv[i + 1])) {
          this->N = 1;
        }
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
    if (this->N != 1) {
      this->N = rand() % 2;
    }
    // if N is true then generate value for A to sum it later in whole group
    if (this->N) {
      this->A = Util::generateDoubleValue(min, max);
      this->group = 1;
    } else {
      this->group = MPI_UNDEFINED;
    }
  }

  void run() {
    // splitting process into different groups
    MPI_Comm_split(MPI_COMM_WORLD, this->group, this->rank, &this->comm);

    // starting timer
    this->startTimer();
    if (this->N) {
      // getting new rank of the process
      MPI_Comm_rank(this->comm, &this->comm);
      // waiting for other processes
      MPI_Barrier(this->comm);
      // getting group size
      MPI_Comm_size(this->comm, &this->groupSize);

      // calling MPI_Allreduce to count sum of A's for processes with N == true
      double sum = 0;
      MPI_Allreduce(&this->A, &sum, 1, MPI_DOUBLE, MPI_SUM, this->comm);
      std::cout << "Process " << this->rank << " (gRank: " << this->groupRank << ")"
                << " result sum: " << sum << " [N: " << this->N << ", "
                << "A: " << this->A << "]" << std::endl;
    } else {
      std::cout << "Process " << this->rank << " (gRank: " << this->groupRank << ") wasn't summing [N = " << this->N
                << "]";
    }
    // ending timer
    this->endTimer();

    // calculating mean time (actual mean will count process 0 later)
    double elapsed_time = this->getElapsedTime();
    double mean_time;
    // reducing time to process 0 of the group (group MPI_UNDEFINED not counting)
    if (this->N) {
      MPI_Reduce(&elapsed_time, &mean_time, 1, MPI_DOUBLE, MPI_SUM, 0, this->comm);
    }
    // process with rank 0 will calculate mean time and put it in file
    if (this->groupRank == 0) {
      // calculating actual mean time
      mean_time = mean_time / this->groupSize;

      this->output("lab-5.output.csv", {{"mean_time", std::to_string(mean_time)}});
      std::cout << "\nTotal elapsed time: " << mean_time;
    }

    MPI_Finalize();
  }
};

// LR6
// FIXME: сделать GroupProcess как родителя. Пока что это делать было впадлу
class TopologyProcess : public Process {
private:
  double A; // value to sum
protected:
  MPI_Comm comm, subComm;
  int commRank, commSize, cartCoords[3];
  int subCommRank, subCommSize;
  // Topology parameters
  int dims[3]{2, 2, -1};
  int periods[3]{true, true, true};
  bool reorder = false;

public:
  TopologyProcess(int argc, char *argv[]) : Process(argc, argv) {
    // If total processes not 8 or 12 then throw exception
    if (this->totalProcesses != 8 && this->totalProcesses != 12) {
      throw std::runtime_error("ERROR: Wrong number of processes. It can be ONLY 8 or 12.");
      MPI_Finalize();
    }

    // Set third dimension size
    this->dims[2] = this->totalProcesses / 4;

    // Creating 3D topology
    MPI_Dims_create(this->totalProcesses, 3, this->dims);
    int cartErr = MPI_Cart_create(MPI_COMM_WORLD, 3, this->dims, this->periods, this->reorder, &this->subComm);
    if (cartErr != MPI_SUCCESS) {
      throw std::runtime_error("ERROR: MPI_Cart_create failed.");
    }

    // rank and coords of process
    MPI_Comm_rank(this->subComm, &this->commRank);
    MPI_Cart_coords(this->subComm, this->commRank, 3, this->cartCoords);

    // Generating 'A' value
    srand(time(0) * rank * rank);
    this->A = Util::generateDoubleValue();

    int remain_dims[3]{1, 1, 0};
    MPI_Cart_sub(this->subComm, remain_dims, &this->subComm);

    // Subgroup rank and size
    MPI_Comm_rank(this->subComm, &this->subCommRank);
    MPI_Comm_size(this->subComm, &this->subCommSize);
  }
  void run() {
    this->startTimer();
    // Print position in 3D grid
    printf("[Process %d] I am located at (%d, %d, %d) with value A = %f.\n", this->subCommRank, this->cartCoords[0],
           this->cartCoords[1], this->cartCoords[2], this->A);

    double subSum = 0.0;
    MPI_Allreduce(&this->A, &subSum, 1, MPI_DOUBLE, MPI_SUM, this->subComm);
    printf("[Subgroup %d, process %d] Sum of values in matrix: %f\n", this->cartCoords[2], this->subCommRank, subSum);
    this->endTimer();

    // calculating mean time
    double mean_time = this->calculateMeanElapsedTime(this->subComm);
    // process with rank 0 will calculate mean time and put it in file
    if (this->subCommRank == 0) {
      this->output("lab-6.output.csv", {{"mean_time", std::to_string(mean_time)}});
      printf("[Subgroup %d, process %d] Mean elapsed time: %f\n", this->cartCoords[2], this->subCommRank, mean_time);
    }

    MPI_Finalize();
  }
};

// LR7
class MatrixProcess : public Process {

private:
  // matrix properties
  int matrixSize = -1;       // initial matrix size
  int paddedMatrixSize = -1; // matrix size to make equal blocks of sub matrices
  std::vector<std::vector<double>> A, B, C, localA, localB, localC;
  int procNum, blockSize;

  // Topology propertiee
  MPI_Comm gridComm;
  int commRank, commSize, coords[2];
  // Topology parameters
  int dims[2]{0, 0};
  int periods[2]{true, true};
  bool reorder = false;

  // what the fuck is this returning type... (please don't beat me up for this, I already did it myself)
  // Returning matrix of matrices total of rows * cols which looks like this:
  // [matrix00, matrix01;
  //  matrix10, matrix11]
  // And each matrixYY contains 2D matrix of double values:
  // [value00, value01;
  //  value10, value11]
  // omfg, this is terrible...
  std::vector<std::vector<std::vector<std::vector<double>>>> splitMatrix(std::vector<std::vector<double>> matrix,
                                                                         int rows, int cols) {
    // Getting sizes for submatrices
    int rowSize = matrix.size() / rows;
    int colSize = matrix.size() / cols;

    // Result matrix of matrices
    std::vector<std::vector<std::vector<std::vector<double>>>> matrices(rows);
    for (int row = 0; row < rows; row++) {
      for (int col = 0; col < cols; col++) {
        std::vector<std::vector<double>> newMatrix(rowSize, std::vector<double>(colSize));
        for (int i = 0; i < rowSize; i++) {
          for (int j = 0; j < colSize; j++) {
            // Skipping some values to distribute all values to right
            newMatrix[i][j] = matrix[i + rowSize * row][j + colSize * col];
          }
        }
        matrices[row].push_back(newMatrix);
      }
    }
    return matrices;
  }

  void padMatrix(std::vector<std::vector<double>> &matrix, int padSize) {
    if (this->matrixSize == padSize) {
      return;
    }

    // resizing first dimension
    matrix.resize(padSize);

    // Resizing second dimension
    for (int i = 0; i < padSize; i++) {
      matrix[i].resize(padSize);
    }
    this->paddedMatrixSize = padSize;
  }

  // Method to multiply local A and B matrices
  std::vector<std::vector<double>> multiplyBlocks(std::vector<std::vector<double>> &A,
                                                  std::vector<std::vector<double>> &B) {
    int size = A.size();
    std::vector<std::vector<double>> C(size, std::vector<double>(size, 0.0));
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        for (int k = 0; k < size; k++) {
          C[i][j] += A[i][k] * B[k][j];
        }
      }
    }
    return C;
  }

  // Cutting matrix to needed size
  // Used when got final C matrix and need to cut empty fields
  std::vector<std::vector<double>> cutMatrix(std::vector<std::vector<double>> matrix, int size) {
    std::vector<std::vector<double>> newMatrix(size, std::vector<double>(size));
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        newMatrix[i][j] = matrix[i][j];
      }
    }
    return newMatrix;
  }

  void printMatrix(std::vector<std::vector<double>> matrix) {
    for (int i = 0; i < matrix.size(); i++) {
      for (int j = 0; j < matrix[i].size(); j++) {
        std::cout << matrix[i][j] << " ";
      }
      std::cout << "\n";
    }
  }

  std::string vectorToString(std::vector<double> vector) {
    std::string stringVector = "";
    for (int i = 0; i < vector.size(); i++) {
      double value = vector[i];
      value = value + 0.5 - (value < 0);
      stringVector += std::to_string((int)value) + " ";
    }
    return stringVector;
  }

  static std::string matrixToString(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty()) return "";

    // Вычисление ширины каждого столбца
    std::vector<size_t> columnWidths;
    size_t numCols = matrix[0].size();
    columnWidths.resize(numCols, 0);

    for (const auto& row : matrix) {
      for (size_t j = 0; j < row.size(); ++j) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << row[j];
        columnWidths[j] = std::max(columnWidths[j], oss.str().size());
      }
    }

    // Форматирование строки
    std::ostringstream formattedMatrix;
    for (const auto& row : matrix) {
      for (size_t j = 0; j < row.size(); ++j) {
        // FIXME: change SETPRECISION to not zero after
        formattedMatrix << std::setw(columnWidths[j]) 
                        << std::fixed << std::setprecision(0) 
                        << row[j] 
                        << " ";
      }
      formattedMatrix << "\n";
    }

    return formattedMatrix.str();
  }
  std::vector<double> matrixToVector(std::vector<std::vector<double>> matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<double> vectorizedMatrix(rows * cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        vectorizedMatrix[j + i * cols] = matrix[i][j];
      }
    }
    return vectorizedMatrix;
  }

  std::vector<std::vector<double>> vectorToMatrix(std::vector<double> vector, int rows, int cols) {
    std::vector<std::vector<double>> unvectorizedMatrix(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        unvectorizedMatrix[i][j] = vector[j + i * cols];
      }
    }
    return unvectorizedMatrix;
  }
  // Turns vector which containt multiple localC matrices to proper C matrix
  // Expample: 
  //  vector = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
  //  will be turned to matrix:
  //    1  2  5  6
  //    3  4  7  8
  //    9  10 13 14
  //    11 12 15 16
  std::vector<std::vector<double>> collectMatrices(std::vector<double> vector) {
    int blocksNum = this->procNum;   // size of processors grid in one dimension (sqrt from processors number)
    int blockSize = this->blockSize; // block size of each small block of the matrix
    std::vector<std::vector<double>> collectedMatrix(this->paddedMatrixSize,
                                                     std::vector<double>(this->paddedMatrixSize));

    int vectorIndex = 0;
    for (int rectRow = 0; rectRow < blocksNum; rectRow++) {
      for (int rectCol = 0; rectCol < blocksNum; rectCol++) {
        for (int row = blockSize * rectRow; row < blockSize + blockSize * rectRow; row++) {
          for (int col = blockSize * rectCol; col < blockSize + blockSize * rectCol; col++) {
            collectedMatrix[row][col] = vector[vectorIndex];
            vectorIndex++;
          }
        }
      }
    }
    return collectedMatrix;
  }
  // FOR TESTING PURPOSES ONLY
  std::vector<std::vector<double>> subtractMatrices(std::vector<std::vector<double>> A, std::vector<std::vector<double>> B) {
    int rows = A.size();
    int cols = A[0].size();
    std::vector<std::vector<double>> C(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        C[i][j] = A[i][j] - B[i][j];
    return C;
  }
public:
  MatrixProcess(int argc, char *argv[]) : Process(argc, argv) {
    for (int i = 0; i < argc; i++) {
      std::string argument = argv[i];
      if (argument == "--size") {
        this->matrixSize = std::atoi(argv[i + 1]);
      }
    }

    // checking if matrix size was presented, otherwise raise exception
    if (this->matrixSize <= 1) {
      throw std::runtime_error("Please type valid matrix size that will be generated with argument '--size "
                               "some_value'. It should be more than 1.");
    }

    // Getting sqrt of all processes to create grid for matrix multiplication
    this->procNum = static_cast<int>(std::sqrt(this->totalProcesses));
    if (this->procNum * this->procNum != this->totalProcesses) {
      throw std::runtime_error("Number of processes should be perfect square (n * n)");
    }

    // Settings dims for future topology
    MPI_Dims_create(this->totalProcesses, 2, this->dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, this->dims, this->periods, 1, &this->gridComm);

    // Getting properties of topology
    MPI_Comm_rank(this->gridComm, &this->commRank);
    MPI_Cart_coords(this->gridComm, this->commRank, 2, this->coords);

    // If matrix size can't be divided to number of processes we are adding empty data to it to be able to be divided
    // by number of processes
    // ALSO changing blockSize to match paddedMatrixSize
    if (this->matrixSize % this->procNum != 0) {
      this->paddedMatrixSize = this->matrixSize + (this->procNum - this->matrixSize % this->procNum);
      this->blockSize = this->paddedMatrixSize / this->procNum;
    }
    else {
      this->paddedMatrixSize = this->matrixSize;
      this->blockSize = this->matrixSize / this->procNum;
    }
  }
  void printStringFromProcess(std::string title, std::string message) {
    std::cout << "\n[ Process " << this->rank + 1 << " ] " << title << "\n" << message;
  }
  void run() {
    // Generating matrix in process 0
    if (this->rank == 0) {
      this->A = Util::generateLinearMatrix(this->matrixSize);
      this->B = Util::generateLinearMatrix(this->matrixSize, true);
      this->C = std::vector<std::vector<double>>(this->matrixSize, std::vector<double>(this->matrixSize, 0));

      if (this->paddedMatrixSize != this->matrixSize) {
        this->padMatrix(this->A, this->paddedMatrixSize);
        this->padMatrix(this->B, this->paddedMatrixSize);
      }
      this->printStringFromProcess("Matrix A:", this->matrixToString(this->A));
      this->printStringFromProcess("Matrix B:", this->matrixToString(this->B));
    }

    // Creating empty local matrices for each process with size of smaller blocks
    this->localA = std::vector<std::vector<double>>(this->blockSize, std::vector<double>(this->blockSize));
    this->localB = std::vector<std::vector<double>>(this->blockSize, std::vector<double>(this->blockSize));
    this->localC = std::vector<std::vector<double>>(this->blockSize, std::vector<double>(this->blockSize, 0));

    // Splitting matrices A and B for further sending to other processes
    auto blocksA = this->splitMatrix(this->A, this->procNum, this->procNum);
    auto blocksB = this->splitMatrix(this->B, this->procNum, this->procNum);

    // Process 0 sends sub matrices to other processes
    int sendRecvSize = this->blockSize * this->blockSize;
    if (this->rank == 0) {
      for (int i = 0; i < procNum; i++) {
        for (int j = 0; j < procNum; j++) {
          // if sub matrix is for process 0, just leave it here, otherwise send them to other processes
          if (i == 0 && j == 0) {
            this->localA = blocksA[i][j];
            this->localB = blocksB[i][j];
          } else {
            // Making vectors out of matrix and send them to correcponding processes
            int destination = i * this->procNum + j;
            std::vector<double> flatBlockA = this->matrixToVector(blocksA[i][j]);
            std::vector<double> flatBlockB = this->matrixToVector(blocksB[i][j]);
            MPI_Send(flatBlockA.data(), sendRecvSize, MPI_DOUBLE, destination, 0, this->gridComm);
            MPI_Send(flatBlockB.data(), sendRecvSize, MPI_DOUBLE, destination, 1, this->gridComm);
          }
        }
      }
    } else { // Other processes are begin to receiving matrices A and B
      std::vector<double> flatLocalA(sendRecvSize);
      std::vector<double> flatLocalB(sendRecvSize);

      // Receiving flat matrices
      MPI_Recv(flatLocalA.data(), sendRecvSize, MPI_DOUBLE, 0, 0, this->gridComm, MPI_STATUS_IGNORE);
      MPI_Recv(flatLocalB.data(), sendRecvSize, MPI_DOUBLE, 0, 1, this->gridComm, MPI_STATUS_IGNORE);

      // Turning flat matrices to actual matrices from vector
      this->localA = this->vectorToMatrix(flatLocalA, this->blockSize, this->blockSize);
      this->localB = this->vectorToMatrix(flatLocalB, this->blockSize, this->blockSize);
    }

    // First initialization shift
    int src, dest;
    
    std::vector<double> flatLocalA = this->matrixToVector(this->localA);
    std::vector<double> flatLocalB = this->matrixToVector(this->localB);
    if (this->coords[0] > 0) {
      MPI_Cart_shift(this->gridComm, 1, -this->coords[0], &src, &dest);
      MPI_Sendrecv_replace(flatLocalA.data(), sendRecvSize, MPI_DOUBLE, dest, 0, src, 0, this->gridComm, MPI_STATUS_IGNORE);
    }
    this->localA = this->vectorToMatrix(flatLocalA, this->blockSize, this->blockSize);

    if (this->coords[1] > 0) {
      MPI_Cart_shift(this->gridComm, 0, -this->coords[1], &src, &dest);
      MPI_Sendrecv_replace(flatLocalB.data(), sendRecvSize, MPI_DOUBLE, dest, 1, src, 1, this->gridComm, MPI_STATUS_IGNORE);
    }
    this->localB = this->vectorToMatrix(flatLocalB, this->blockSize, this->blockSize);

    MPI_Barrier(this->gridComm);

    for (int step = 0; step < this->procNum; step++) {
      // Sending A matrix vertically up
      // std::vector<double> flatLocalA = this->matrixToVector(this->localA);
      MPI_Cart_shift(this->gridComm, 1, -1, &src, &dest);
      MPI_Sendrecv_replace(flatLocalA.data(), sendRecvSize, MPI_DOUBLE, dest, 0, src, 0, this->gridComm, MPI_STATUS_IGNORE);
      this->localA = this->vectorToMatrix(flatLocalA, this->blockSize, this->blockSize);

      // Sending B matrix horizontally to the left
      // std::vector<double> flatLocalB = this->matrixToVector(this->localB);
      MPI_Cart_shift(this->gridComm, 0, -1, &src, &dest);
      MPI_Sendrecv_replace(flatLocalB.data(), sendRecvSize, MPI_DOUBLE, dest, 1, src, 1, this->gridComm, MPI_STATUS_IGNORE);
      this->localB = this->vectorToMatrix(flatLocalB, this->blockSize, this->blockSize);

      // Calculating C matrix
      auto tempC = this->multiplyBlocks(this->localA, this->localB);
      for (int i = 0; i < this->blockSize; i++) {
        for (int j = 0; j < this->blockSize; j++) {
          this->localC[i][j] += tempC[i][j];
        }
      }
    }

    // Gathering all C matrices
    std::vector<double> flatLocalC = this->matrixToVector(this->localC);
    std::vector<double> flatC(this->paddedMatrixSize * this->paddedMatrixSize);
    MPI_Gather(flatLocalC.data(), flatLocalC.size(), MPI_DOUBLE, flatC.data(), flatLocalC.size(), MPI_DOUBLE, 0,
               this->gridComm);

   // Turning vector to a matrix again
    if (this->rank == 0) {
      this->C = this->collectMatrices(flatC);
      if (this->matrixSize != this->paddedMatrixSize) {
        this->C = this->cutMatrix(this->C, this->matrixSize);
      }

        this->printStringFromProcess("Result matrix C:", this->matrixToString(this->C));
      std::vector<std::vector<double>> standartMultiplicationC = this->multiplyBlocks(this->A, this->B);
      this->printStringFromProcess("Matrix C with default multiplication: ", this->matrixToString(standartMultiplicationC));
    }
    //FIXME: Cut extra elements in the matrix
  }
};

