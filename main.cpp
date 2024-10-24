#include "my_process.cpp"

int main(int argc, char *argv[]) {
  Process *process = nullptr;
  for (int i = 0; i < argc; i++) {
    std::string argument = argv[i];
    if (argument == "--lab") {
      std::string labNumber = argv[i + 1];

      if (labNumber == "1") process = new SimpleProcess(argc, argv);
      if (labNumber == "2") process = new VectorProcess(argc, argv);
      if (labNumber == "3") process = new NetworkProcess(argc, argv);
      if (labNumber == "4") process = new CollectiveProcess(argc, argv);
      if (labNumber == "5") process = new GroupProcess(argc, argv);
      if (labNumber == "6") process = new TopologyProcess(argc, argv);
      if (labNumber == "7") process = new MatrixProcess(argc, argv);
    }
  }

  if (!process)
    throw std::runtime_error("No lab number presented. Pass '--lab value' in the terminal to launch one of the laboratory works");

  process->run();
}
