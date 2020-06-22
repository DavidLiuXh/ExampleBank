#include <pthread.h>
#include <unistd.h>

#include <iostream>
#include <atomic>

void *thread_fun(void *arg)  {
  int i = 1;  
  std::cout << "thread start" << std::endl;  
  while(1)  {
    i++;  
    pthread_testcancel();
  }

  return (void *)1;  
}

// 测试 pthread_cancel
void test_pthread_cancel() {
  void* ret = NULL;  
  pthread_t tid;  
  pthread_create(&tid,NULL,thread_fun,NULL);  
  sleep(1);  

  pthread_cancel(tid);//取消线程  
  pthread_join(tid, &ret);  
  std::cout << "thread exit code: " << reinterpret_cast<long>(ret) << std::endl;
}

std::atomic<bool> kStop(false);
void *thread_detach_fun(void *arg)  {
  pthread_detach(pthread_self());

  std::cout << "thread start" << std::endl;  

#if 1 
  while(1)  {
#else
  //while(kStop)  {
#endif
    sleep(1);
#if 0
    pthread_testcancel();
#endif
  }

  std::cout << "thread end" << std::endl;  

  return (void *)1;  
}

void test_detach_thread() {
  pthread_t tid;  
  pthread_create(&tid, NULL, thread_detach_fun, NULL);  
  sleep(2);  

  kStop = true;
  //pthread_cancel(tid);//取消线程  
  //void* ret = NULL;
  //pthread_join(tid, &ret);  
  std::cout << "main thread end" << std::endl;  
}

int main(int argc, char* argv[]) {
  //test_pthread_cancel();

  test_detach_thread();

  getchar();

  return 0;
}
