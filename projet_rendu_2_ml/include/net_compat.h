#ifndef NET_COMPAT_H
#define NET_COMPAT_H
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET socket_t;
#define CLOSESOCK closesocket
static int net_init(void){WSADATA w;return WSAStartup(MAKEWORD(2,2),&w)==0;}
static void net_cleanup(void){WSACleanup();}
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
typedef int socket_t;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define CLOSESOCK close
static int net_init(void){return 1;}
static void net_cleanup(void){}
#endif
#endif
