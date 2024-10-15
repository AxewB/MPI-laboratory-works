#include "my_process.cpp"


int main(int argc, char* argv[]) {
  // SimpleProcess *sp = new SimpleProcess(argc, argv);
  // sp->run();

  VectorProcess *vp = new VectorProcess(argc, argv);
  vp->run(); 

  // NetworkProcess *np = new NetworkProcess(argc, argv);
  // np->run();

  // Process *process = new NumbersProcess(argc, argv);
  // process->run();

  // Process *process = new GroupProcess(argc, argv);
  // process->run();
}  

  