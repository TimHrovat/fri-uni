
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "naloga1.h"

char *zdruzi(char **nizi, char *locilo) {
  char **strings = nizi;
  int len = 1;
  int locilo_len = strlen(locilo);

  while (*strings != NULL) {
    len += strlen(*strings);

    if (*(++strings) != NULL) {
      len += locilo_len;
    }
  }

  strings = nizi;
  char *final_string = (char *)malloc(sizeof(char) * len);
  char *str = final_string;

  while (*strings != NULL) {
    strcpy(str, *strings);

    str += strlen(*strings);

    if (*(++strings) != NULL) {
      strcpy(str, locilo);

      str += locilo_len;
    }
  }

  *str = '\0';

  return final_string;
}

#ifndef test

int main() {
  // koda za ro"cno testiranje
  return 0;
}

#endif

0000000000000001

0000:1800:0600:0000:0060:0018:0006:0001

0:1800:600::60:18:6:1
