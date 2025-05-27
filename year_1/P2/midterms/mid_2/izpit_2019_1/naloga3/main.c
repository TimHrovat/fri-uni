#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  //
  int n = strtol(argv[1], NULL, 10);
  int count = 1;
  int offset = 0;
  int st_vrhov;

  do {
    st_vrhov = n / 2 + offset;
    int st_dolin = st_vrhov - 1;

    int st_moznosti = pow(2, st_dolin) - 1;

    count += st_moznosti;

    offset--;
  } while (st_vrhov > 1);

  printf("%d", count);

  return 0;
}
