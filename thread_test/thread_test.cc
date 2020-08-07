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

void *thread_loop_func(void *arg)  {
  pthread_detach(pthread_self());

  std::cout << "thread start" << std::endl;  

  while(1)  {
    usleep(100);
  }

  std::cout << "thread end" << std::endl;  

  return (void *)1;  
}

//查看：ps  -T -eo tid,pid,ppid,pgid,tgid,comm | grep thread_test
//可以查看包括线程
void TestPidPgid() {
  fork();

  while(1);
}

void TestThread() {
  //test_pthread_cancel();

  //test_detach_thread();

  //getchar();
  //
  pthread_t tid;  
  pthread_create(&tid, NULL, thread_loop_func, NULL);  
  sleep(2);  

  //这个pthread_exit()用在main thread里的时候，
  //它将使这个main thread退出，但是允许其他线程继续执行，直到结束。
  //在其他线程运行过程中，从top中可以看到这个main thread状态是z。
  //另外还在运行的线程状态是R。
  //这个main thread是z状态，是一种假的僵尸进程，此时使用kill -9 pid可以杀死，真的僵尸进程 kill -9 无法杀死。
  //为什么这里可以杀死呢？因为main thread tid正好是这个进程组id

  //pthread_exit(0);
}

int main(int argc, char* argv[]) {
  TestThread();
  TestPidPgid();

  return 0;
}
