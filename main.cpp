#include "my_process.cpp"

int main(int argc, char *argv[]) {
  Process *process = nullptr;
  for (int i = 0; i < argc; i++) {
    if (argv[i] == "--lab") {
      if (argv[i + 1] == "1") process = new SimpleProcess(argc, argv);
      if (argv[i + 1] == "2") process = new VectorProcess(argc, argv);
      if (argv[i + 1] == "3") process = new NetworkProcess(argc, argv);
      if (argv[i + 1] == "4") process = new NumbersProcess(argc, argv);
      if (argv[i + 1] == "5") process = new TopologyProcess(argc, argv);
      if (argv[i + 1] == "6") process = new MatrixProcess(argc, argv);
    }
  }

  if (!process)
    throw std::runtime_error("No lab number presented. Pass '--lab value' in the terminal to launch one of the laboratory works");

  process->run();
}
