#include "my_process.cpp"

int main(int argc, char* argv[]) {
  SimpleProcess *sp = new SimpleProcess(argc, argv, true, false);
  sp->run();
} 