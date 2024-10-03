#include <mpi.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>

/**
 * Два варианта решения:?
 * 1. Разделять просто количество элементов на количество процессов. 
 *  - Т.е. если в векторе 10 элементов, то 10 процессов должны обработать умножение по одному элементу. После выполнения отчитываются перед родителем
 *  - Если ЭЛЕМЕНТОВ меньше, чем ПРОЦЕССОВ, то некоторые процессы не задействуются
 *  - Если ПРОЦЕССОВ меньше, чем ЭЛЕМЕНТОВ, то элементы распределяются неравномерно
 * 2. Двоичное дерево. Делим векторы пополам - отсылаем на дочерние процессы. 
 * 
 * Если захочу сделать для n векторов
 * - про классы и функции (произвольное число векторов в качестве параметра): https://www.perplexity.ai/search/kak-mne-sdelat-klass-i-unasled-B.zg97FeRAaU9baIB4Tnkw
 * - про скалярное произведение n векторов: https://www.perplexity.ai/search/kak-vygliadit-skaliarnoe-proiz-THimtztbTYaPJXQotf3dcw
 */


class Process {
private:
  int totalProcesses = NULL;
  int rank = NULL;
public:
  Process() {
    MPI_Comm_size(MPI_COMM_WORLD, &this->totalProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
  }
  int getTotalProcesses() { return totalProcesses; }
  int setTotalProcesses(int totalProcesses) { this->totalProcesses = totalProcesses; }
  int getRank() { return rank; }
  int setRank(int rank) { this->rank = rank; }
};

class VectorProcess : public Process {
private:
  int *vectorA = nullptr;
  int *vectorB = nullptr;
  int vectorSize = NULL;
public:
  VectorProcess() : Process() { }
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
  void run() {
    init();


    // TODO: main method in algorithm
  }
};

// Что если количество элементов в векторе четное
// Что если количество элементов в векторе нечетное
// Что если количество процессов четное
// Что если количество процессов нечетное
// Что если количество процессов всего 2 (основной и дополнительный)
// Что если количество процессов больше, чем количество элементов в векторах
// Как разделять векторы на каждый процесс
int multiplyVectors(int a[], int b[]) {
  int size = sizeof(a) / sizeof(a[0]);
  int result = 0;
  for (int i = 0; i < size; i++) {
    result += a[i] * b[i];
  }
  return result;
}

int *inputVector(int size) {
  int *result = new int[size];
  for (int i = 0; i < size; i++)
  {
    try
    {
      std::cin >> result[i];
    }
    catch (const std::exception &e)
    {
      std::cout << "Invalid input. Please enter an integer." << std::endl;
      return nullptr;
    }
  }
  return result;
}

int main(int argc, char *argv[]) {
  // MPI variables
  int procRank, size;

  // vector related variables
  int vectorSize = NULL;
  int *vectorA = nullptr;
  int *vectorB = nullptr;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank); // getting rank of current process
  MPI_Comm_size(MPI_COMM_WORLD, &size);     // getting total number of processes

  // input vector size
  while (true) {
    std::cout << "Enter the size of the vectors (even number): ";
    try {
      std::cin >> vectorSize;
      break;
    }
    catch (const std::exception &e) {
      std::cout << "Invalid input. Please enter an integer." << std::endl;
      continue;
    }
  }

  // input vectors
  while (vectorA == nullptr || vectorB == nullptr) {
    vectorA = inputVector(vectorSize);
    vectorB = inputVector(vectorSize);
  }

  MPI_Finalize();
  return 0;
}