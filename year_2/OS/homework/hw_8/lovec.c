#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

volatile sig_atomic_t energy = 42;
volatile sig_atomic_t symbol = 0;

void handle_sigterm(int sig);
void handle_sigusr1(int sig);
void handle_sigusr2(int sig);
void handle_sigchld(int sig);

int main(int argc, char *argv[]) {
  if (argc > 1) {
    energy = atoi(argv[1]);
  }

  printf("My PID: %d\n", getpid());
  printf("Starting with energy: %d.\n", energy);
  fflush(stdout);

  signal(SIGTERM, handle_sigterm);
  signal(SIGUSR1, handle_sigusr1);
  signal(SIGUSR2, handle_sigusr2);
  signal(SIGCHLD, handle_sigchld);

  while (energy > 0) {
    putchar(symbol ? '*' : '.');
    fflush(stdout);
    energy--;
    sleep(1);
  }

  printf("\nOut of energy. Aggghnhhrrrrr.\n");
  return 0;
}

void handle_sigterm(int sig) {
  energy += 10;
  printf("\n***Yahoo! Bonus energy (%d).\n", energy);
  fflush(stdout);
}

void handle_sigusr1(int sig) {
  symbol = !symbol;
  printf("\n**Symbol toggled to %c.\n", symbol ? '*' : '.');
  fflush(stdout);
}

void handle_sigusr2(int sig) {
  pid_t pid = fork();
  if (pid == 0) {
    int sleep_time = (energy % 7) + 1;
    sleep(sleep_time);
    int exit_status = (42 * energy) % 128;
    printf("****Child exit with status: %d.\n", exit_status);
    fflush(stdout);
    exit(exit_status);
  } else if (pid > 0) {
    printf("****Forked child %d.\n", pid);
    fflush(stdout);
  }
}

void handle_sigchld(int sig) {
  int status;
  pid_t pid;
  while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
    if (WIFEXITED(status)) {
      printf("*Zombie caught with status: %d\n", WEXITSTATUS(status));
      fflush(stdout);
    }
  }
}
