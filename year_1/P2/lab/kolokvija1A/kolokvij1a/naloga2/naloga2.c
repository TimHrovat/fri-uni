
/*
 * Prevajanje in zagon testnega programa testXY.c:
 *
 * gcc -D=test testXY.c naloga2.c
 * ./a.out
 *
 * Zagon testne skripte ("sele potem, ko ste prepri"cani, da program deluje!):
 *
 * export name=naloga2
 * make test
 *
 * Javni testni primeri (po te"zavnosti):
 * -- 02, 03, 04: en sam vhodni niz;
 * -- 05, 06, 07: vsi vhodni nizi so enako dolgi;
 * -- 01, 08, 09, 10: splo"sni primeri.
 *
 * Javni testni primeri (po na"cinu priprave):
 * -- 01: primer iz besedila;
 * -- 01, 02, 05: ro"cno ustvarjeni;
 * -- ostali: samodejno generirani.
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// po potrebi dopolnite ...

//=============================================================================

char **poStolpcih(char **nizi, int stVhodnih, int *stIzhodnih) {
  int max_len = 0;
  for (int i = 0; i < stVhodnih; i++) {
    int len = strlen(nizi[i]);

    if (len > max_len) {
      max_len = len;
    }
  }
  *stIzhodnih = max_len;

  char **tabela = calloc(max_len, sizeof(char **));
  for (int i = 0; i < max_len; i++) {
    tabela[i] = calloc(stVhodnih + 1, sizeof(char *));
    char *ptr_str = tabela[i];

    for (int j = 0; j < stVhodnih; j++) {
      if (strlen(nizi[j]) > i) {
        *ptr_str = nizi[j][i];
        ptr_str++;
      }

      *ptr_str = '\0';
    }
  }

  return tabela;
}

//=============================================================================

#ifndef test

char *NIZI[] = {
    "Danes",
    "je",
    "kolokvij",
    "pri_P2!",
};

int main() {
  int stIzhodnih = 0;
  char **rezultat =
      poStolpcih(NIZI, sizeof(NIZI) / sizeof(NIZI[0]), &stIzhodnih);
  printf("%d\n", stIzhodnih);

  for (int i = 0; i < stIzhodnih; i++) {
    printf("%s\n", rezultat[i]);
    free(rezultat[i]);
  }
  free(rezultat);

  return 0;
}

#endif
