#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  FILE *file = fopen(argv[1], "rb");
  int row = strtol(argv[2], NULL, 10);
  int col = strtol(argv[3], NULL, 10);
  char buf[1000];

  int max_row, max_col;
  fscanf(file, "%s", buf);
  fscanf(file, "%d %d", &max_col, &max_row);
  fscanf(file, "%s", buf);
  fgetc(file);

  int px_index = (max_col * row + col) * 3;

  fseek(file, px_index, SEEK_CUR);

  unsigned char px[3];

  fread(px, sizeof(unsigned char), 3, file);

  printf("%d %d %d\n", px[0], px[1], px[2]);

  fclose(file);
  return 0;
}
