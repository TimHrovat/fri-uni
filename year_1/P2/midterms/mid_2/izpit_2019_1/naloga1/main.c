#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool str_includes(char *str, char chr);

int main() {
  char *f_in = (char *)calloc(21, sizeof(char));
  char *f_out = (char *)calloc(21, sizeof(char));
  char chr;

  scanf("%s", f_in);
  scanf("%s", f_out);
  scanf(" %c", &chr);

  FILE *fptr_in = fopen(f_in, "r");
  FILE *fptr_out = fopen(f_out, "w");

  char line_buf[1000];
  char last_line_with_chr[1000];

  while (fgets(line_buf, 1000, fptr_in)) {
    if (str_includes(line_buf, chr)) {
      strcpy(last_line_with_chr, line_buf);
      last_line_with_chr[strlen(line_buf)] = '\0';
    }
  }

  fprintf(fptr_out, "%s", last_line_with_chr);

  fclose(fptr_in);
  fclose(fptr_out);

  return 0;
}

bool str_includes(char *str, char chr) {
  int len = strlen(str);

  for (int i = 0; i < len - 1; i++) {
    if (str[i] == chr) {
      return true;
    }
  }

  return false;
}
