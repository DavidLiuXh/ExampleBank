#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>

int main(int argc, char *argv[]) {
  int sockfd = -1;

  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (-1 == sockfd) {
    perror( "sock created" );
    exit( -1 );
  }

  struct sockaddr_in server;
  memset( &server, 0, sizeof( struct sockaddr_in ) );
  server.sin_family = AF_INET;
  server.sin_port = htons(9999);
  server.sin_addr.s_addr = inet_addr("10.19.100.28");

  int res = -1;
  res = connect( sockfd, (struct sockaddr*)&server, sizeof( server ) );
  if( -1 == res ) {
    perror( "sock connect" );
    exit( -1 );
  }

  int flags = fcntl(sockfd, F_GETFL, 0);
  fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

  getchar();

  char sendBuf[1024] = { 0 };
  //write( sockfd, sendBuf, sizeof( sendBuf ) );
  
  //getchar();

  char recvBuf[1024] = { 0 };
  read( sockfd, recvBuf, sizeof( recvBuf ) );

  //getchar();

  printf("exit...\n");
  while (read( sockfd, recvBuf, sizeof( recvBuf ) ) > 0);
  close(sockfd);

  getchar();

  return 0;
}
