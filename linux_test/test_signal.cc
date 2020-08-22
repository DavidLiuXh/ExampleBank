#include <unistd.h>

#include <ctime>
#include <csignal>

#include <iostream>
#include <thread>

using namespace std::chrono_literals;

void IntHandler(int signum) {
  std::cout << time(NULL) << " Got a int signal" << signum << std::endl;
  std::this_thread::sleep_for(5s);
  std::cout << time(NULL) << " Fininsh int signal" << signum << std::endl;
}

void QuitHandler(int signum) {
  std::cout << time(NULL) << " Got a quit signal" << signum << std::endl;
}

void MinPlus2Handler(int signum) {
  std::cout << time(NULL) << " Got a SIGRTMIN + 2 signal" << signum << std::endl;
}

void TestSignal() {
  signal(SIGINT, IntHandler);
  signal(SIGQUIT, QuitHandler);
  //real-time signal
  signal(SIGRTMIN+2, MinPlus2Handler);

  while(true) {
    std::this_thread::sleep_for(10s);
    std::cout << "." << std::endl;
  }

  std::cout << std::endl;
}

void AlarmHandler(int signum) {
  std::cout << "Got a alarm signal" << std::endl;
}

void TestAlarm() {
  struct sigaction sig;
  sig.sa_handler = AlarmHandler;
  //sig.sa_handler = SIG_IGN;
  sig.sa_flags = 0;
  sigemptyset(&sig.sa_mask);

  struct sigaction old;
  sigaction(SIGALRM, &sig, &old);
  std::cout << "A SIGALRM handler has registered" << std::endl;

  alarm(3);

  pause();

  //sigaction(SIGALRM, &sig, &old);
  std::cout << "Raise another alarm signal, in 2 second later" << std::endl;
  alarm(2);

  std::cout << "Star sleep 10s ..." << std::endl;
#if 0
  //sleep_for不能被信号中断
  std::this_thread::sleep_for(10s);
#else
  //信号发生时，被中断
  sleep(10);
#endif
  std::cout << "Exit sleep 10s ..." << std::endl;

  std::cout << "Exit..." << std::endl;
  sigaction(SIGALRM, &old, NULL);
}
int main(int argc, char* argv[]) {
  TestSignal();
  //TestAlarm();

  return 0;
}
