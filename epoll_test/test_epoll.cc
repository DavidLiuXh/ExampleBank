#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>

#define PORT 9999
#define MAX_EVENTS 10

static int tcp_listen() {
  int lfd, opt, err;
  struct sockaddr_in addr;

  lfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  assert(lfd != -1);

  opt = 1;
  err = setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  assert(!err);

  bzero(&addr, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(PORT);

  err = bind(lfd, (struct sockaddr *)&addr, sizeof(addr));
  assert(!err);

  err = listen(lfd, 8);
  assert(!err);

  return lfd;
}

static void epoll_ctl_add(int epfd, int fd, int evts) {

  struct epoll_event ev;
  ev.events = evts;
  ev.data.fd = fd;
  int err = epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev);
  assert(!err);
}

#define SPLIT_AND_PRINT_EPOLLEVENT(epoll_events, event) \
  do { \
    if (e->events & event) { \
      printf("fired event:%s\n", #event); \
    } \
  } while(0)

static void handle_events(struct epoll_event *e, int epfd) {
  printf("events: socket %d -> events %d\n", e->data.fd, e->events);

  SPLIT_AND_PRINT_EPOLLEVENT(e, EPOLLIN);
  SPLIT_AND_PRINT_EPOLLEVENT(e, EPOLLOUT);
  SPLIT_AND_PRINT_EPOLLEVENT(e, EPOLLERR);
  SPLIT_AND_PRINT_EPOLLEVENT(e, EPOLLHUP);
  SPLIT_AND_PRINT_EPOLLEVENT(e, EPOLLMSG);
  SPLIT_AND_PRINT_EPOLLEVENT(e, EPOLLRDHUP);
  SPLIT_AND_PRINT_EPOLLEVENT(e, EPOLLRDNORM);
  SPLIT_AND_PRINT_EPOLLEVENT(e, EPOLLWRNORM);
  SPLIT_AND_PRINT_EPOLLEVENT(e, EPOLLWRBAND);

  char buff[1024] = {0};

  if (e->events & EPOLLERR) {
    int error = 0;
    socklen_t errlen = sizeof(error);
    if(getsockopt(e->data.fd, SOL_SOCKET, SO_ERROR, (void *)&error, &errlen) == 0) {
      printf("When EPOLLERR | errno:%d | error = %s\n", error, strerror(error));
    }
  }

  if (e->events & EPOLLRDHUP) {
#if 0
    close(e->data.fd);
    printf("RDHUP | peer closed\n");
    return;
#endif
  }

  if (e->events & EPOLLIN) {
#if  1 
    ssize_t data_len = read(e->data.fd, buff, 1024);
    printf("Read data len = %ld | errno: %d\n", data_len, errno);

    //data_len = write(e->data.fd, buff, 1024);
    //printf("Write data len = %ld | errno: %d\n", data_len, errno);

    if (0 == data_len) {
#if 0
      data_len = write(e->data.fd, buff, 1024);
      printf("Write data len = %ld | errno: %d\n", data_len, errno);
#endif
#if 0
      //close(e->data.fd);
      //printf("Close \n");
#endif
    } else if (data_len == -1) {
      perror("failed read | ");
      close(e->data.fd);
      perror("failed close | ");
    }
#endif
  }

  if (e->events & EPOLLOUT) {
#if 0
    printf("Call to write...");
    ssize_t data_len = write(e->data.fd, buff, 1);
    printf("Write data len = %ld | errno:%d\n", data_len, errno);
    perror("Write error | ");
#endif
  }

  /*
  int err = shutdown(e->data.fd, SHUT_WR);
  err = close(e->data.fd);
  if (err) {
    printf("shutdown errno: %d\n", errno);
    exit(123);
  }
  */
}

int main(int argc, char *argv[]) {
  //signal(SIGPIPE, SIG_IGN);

  int epfd, lfd, cfd, n;
  struct epoll_event events[MAX_EVENTS];

  epfd = epoll_create1(0);
  assert(epfd != -1);

  lfd = tcp_listen();
  epoll_ctl_add(epfd, lfd, EPOLLIN);

  for (;;) {
    //sleep(6);
    n = epoll_wait(epfd, events, MAX_EVENTS, -1);

    for (int i = 0; i < n; i++) {
      if (events[i].data.fd == lfd) {
        cfd = accept(lfd, NULL, NULL);
        //epoll_ctl_add(epfd, cfd, EPOLLIN | EPOLLOUT | EPOLLET);
        epoll_ctl_add(epfd, cfd,  EPOLLIN | EPOLLOUT | EPOLLRDHUP);
        //epoll_ctl_add(epfd, cfd,  EPOLLRDHUP);
      } else {
        handle_events(&events[i], epfd);
      }
    }
  }

  return 0;
}
