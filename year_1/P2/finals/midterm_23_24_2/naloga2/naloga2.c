#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char **read_line(FILE *file, int n);
void clear_line(char **line, int n);

int main(int argc, char **argv) {
  FILE *file_in = fopen(argv[1], "r");
  FILE *file_out = fopen(argv[2], "w");
  int m = atoi(argv[3]);
  int out_indexes[argc - 4];
  int out_count = argc - 4;

  char **line = (char **)malloc(sizeof(char *) * m);
  for (int i = 0; i < m; i++) {
    line[i] = calloc(100, sizeof(char));
    fscanf(file_in, "%s", line[i]);
  }

  for (int i = 4; i < argc; i++) {
    for (int j = 0; j < m; j++) {
      if (strcmp(line[j], argv[i]) == 0) {
        out_indexes[i - 4] = j;
        break;
      }
    }
  }

  for (int i = 0; i < out_count; i++) {
    if (i == out_count - 1) {
      fprintf(file_out, "%s\n", line[out_indexes[i]]);
    } else {
      fprintf(file_out, "%s,", line[out_indexes[i]]);
    }
  }

  while (true) {
    if (fscanf(file_in, "%s", line[0]) != 1) {
      break;
    }

    for (int i = 1; i < m; i++) {
      fscanf(file_in, "%s", line[i]);
    }

    for (int i = 0; i < out_count; i++) {
      if (i == out_count - 1) {
        fprintf(file_out, "%s\n", line[out_indexes[i]]);
      } else {
        fprintf(file_out, "%s,", line[out_indexes[i]]);
      }
    }
  }

  fclose(file_in);
  fclose(file_out);

  return 0;
}
