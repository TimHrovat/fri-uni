#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

static int child_i;

void alarm_handler(int sig) { _exit(child_i); }

int main(int argc, char *argv[]) {
  int N = (argc > 1) ? atoi(argv[1]) : 10;
  pid_t children[N];

  for (int i = 0; i < N; i++) {
    pid_t pid = fork();
    if (pid == 0) {
      child_i = i;
      signal(SIGALRM, alarm_handler);
      alarm(42);
      while (1) {
        sleep(1 + child_i);
        putchar('A' + child_i);
        fflush(stdout);
      }
    } else {
      children[i] = pid;
    }
  }

  for (int j = 0; j < 10; j++) {
    sleep(1);
    putchar('*');
    fflush(stdout);
  }

  for (int i = 0; i < N; i++) {
    waitpid(children[i], NULL, 0);
  }

  printf("\nSamo brez panike\n");
  exit(42);
}
