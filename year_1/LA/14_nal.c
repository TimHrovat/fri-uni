#include <stdio.h>
#include <stdlib.h>

long calculate(long long a1, long long a2, long current_index, long max_index) {
  if (max_index == current_index) {
    return a1;
  }
  long next = calculate(5 * a1 - 6 * a2, a1, current_index + 1, max_index);

  return next;
}

int main(int argc, char **argv) {
  int i = atoi(argv[1]);
  printf("%ld\n", calculate(1, 1, 1, i));
  return 0;
}
