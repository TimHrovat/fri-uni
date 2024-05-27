#include <stdio.h>

void obrni(FILE *vhod, FILE *izhod);

int main(int argc, char **argv) {
  FILE *f_in = fopen(argv[1], "r");
  FILE *f_out = fopen(argv[2], "w");

  obrni(f_in, f_out);

  fclose(f_in);
  fclose(f_out);

  return 0;
}

void obrni(FILE *vhod, FILE *izhod) {
  char line[1000];

  if (!fgets(line, 1000, vhod)) {
    return;
  }

  obrni(vhod, izhod);

  fprintf(izhod, "%s", line);
}
