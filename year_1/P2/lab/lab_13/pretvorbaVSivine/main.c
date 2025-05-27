#include <stdio.h>

int main(int argc, char **argv) {
  FILE *file_in = fopen(argv[1], "rb");
  FILE *file_out = fopen(argv[2], "wb");
  int w, h;

  char buf[1000];
  fscanf(file_in, "%s", buf);
  fscanf(file_in, "%d %d", &w, &h);
  fscanf(file_in, "%s", buf);
  getc(file_in);

  fprintf(file_out, "P5\n%d %d\n255\n", w, h);

  int n = w * h;
  unsigned char rgb[3];

  for (int i = 0; i < n; i++) {
    fread(rgb, sizeof(unsigned char), 3, file_in);

    int grey = (30 * rgb[0] + 59 * rgb[1] + 11 * rgb[2]) / 100;

    fwrite(&grey, sizeof(unsigned char), 1, file_out);
  }

  fclose(file_in);
  fclose(file_out);
  return 0;
}
