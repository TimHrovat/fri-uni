#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

volatile bool running = true;

void *thread_function(void *arg);

int main(int argc, char *argv[]) {
  int M = 5;

  int N = atoi(argv[1]);
  if (argc > 2) {
    M = atoi(argv[2]);
  }

  pthread_t threads[N];
  int thread_ids[N];

  for (int i = 0; i < N; i++) {
    thread_ids[i] = i;
    pthread_create(&threads[i], NULL, thread_function, &thread_ids[i]);
  }

  sleep(M);

  running = false;

  for (int i = 0; i < N; i++) {
    pthread_join(threads[i], NULL);
  }

  return 0;
}

void *thread_function(void *arg) {
  int thread_id = *(int *)arg;

  while (running) {
    printf("Nit %d\n", thread_id);
    sleep(1);
  }

  return NULL;
}
