
/*
 * Prevajanje in zagon testnega programa testXY.c:
 *
 * gcc -Dtest testXY.c naloga2.c
 * ./a.out
 *
 * Zagon testne skripte ("sele potem, ko ste prepri"cani, da program deluje!):
 *
 * export name=naloga2
 * make test
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "naloga2.h"

//=============================================================================

// po potrebi dopolnite ...

void razbohoti(char **nizi) {
  char **_nizi = nizi;
  int len = 0;

  while (*_nizi != NULL) {
    len++;
    _nizi++;
  }

  for (int i = 0; i < len; i++) {
    for (int j = i + 1; j < len; j++) {
      if (strcmp(nizi[i], nizi[j]) == 0 && nizi[i] == nizi[j]) {
        nizi[j] = (char *)calloc(strlen(nizi[j]) + 1, sizeof(char));
        strcpy(nizi[j], nizi[i]);
      }
    }
  }
}

//=============================================================================

#ifndef test

int main(int argc, char **argv) {
  // "Ce "zelite funkcijo <razbohoti> testirati brez testnih primerov,
  // dopolnite to funkcijo in prevedite datoteko na obi"cajen na"cin
  // (gcc naloga2.c).
  return 0;
}

#endif
