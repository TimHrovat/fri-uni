#include <stdio.h>

int main(int argv, char **argc) {
  FILE *fptr = fopen("args.txt", "w");

  if (fptr == NULL) {
    return 1;
  }

  for (int i = 0; i < argv; i++) {
    fprintf(fptr, "%s\n", argc[i]);
  }

  fclose(fptr);

  return 0;
}
