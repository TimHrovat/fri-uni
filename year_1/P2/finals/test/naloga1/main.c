#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  char n_str[101];
  int k;

  scanf("%s %d", n_str, &k);

  long long n = atoll(n_str);

  long long mult = n * k;

  printf("%lld\n", mult);

  return 0;
}
