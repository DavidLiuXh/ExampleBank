#include <pthread.h>
#include <unistd.h>

#include <iostream>

void *thread_fun(void *arg)  {
  int i = 1;  
  std::cout << "thread start" << std::endl;  
  while(1)  {
    i++;  
    pthread_testcancel();
  }

  return (void *)0;  
}

void test() {
  void *ret = NULL;  
  int iret=0;  
  pthread_t tid;  
  pthread_create(&tid,NULL,thread_fun,NULL);  
  sleep(1);  

  pthread_cancel(tid);//取消线程  
  pthread_join(tid, &ret);  
  std::cout << "thread exit code: " << ret << std::endl;
}

int main(int argc, char* argv[]) {
  test();

  return 0;
}
