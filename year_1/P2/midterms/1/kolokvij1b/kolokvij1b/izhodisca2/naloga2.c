
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
 * -- 02, 03: dol"zina vsakega vhodnega niza je enaka ciljnaDolzina;
 * -- 04, 05, 06, 07: dol"zina vsakega vhodnega niza je enaka 1;
 * -- 01, 08, 09, 10: splo"sni primeri.
 *
 * Javni testni primeri (po na"cinu priprave):
 * -- 01: primer iz besedila;
 * -- 01, 02, 04: ro"cno ustvarjeni;
 * -- ostali: samodejno generirani.
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// po potrebi dopolnite ...

//=============================================================================

char *strpad(char *str, int left_pad, int right_pad) {
  char *ptr_str = str;
  int str_len = strlen(ptr_str);
  int final_len = str_len + left_pad + right_pad;
  char *padded_str = (char *)malloc(sizeof(char *) * (final_len + 1));
  char *ptr_padded_str = padded_str;

  for (int i = 0; i < final_len; i++) {
    if (i < left_pad) {
      *ptr_padded_str = '.';
    } else if (i < left_pad + str_len) {
      *ptr_padded_str = *ptr_str;
      ptr_str++;
    } else {
      *ptr_padded_str = '.';
    }

    ptr_padded_str++;
  }

  *ptr_padded_str = '\0';

  return padded_str;
}

char **naSredino(char **nizi, int ciljnaDolzina) {
  char **ptr_nizi = nizi;

  int len = 0;
  while (*ptr_nizi != NULL) {
    len++;
    ptr_nizi++;
  }

  ptr_nizi = nizi;
  char **final_strings = malloc(sizeof(char **) * (len + 1));
  char **ptr_final_strings = final_strings;

  for (int i = 0; i < len; i++) {
    int string_len = strlen(*ptr_nizi);
    int diff_len = ciljnaDolzina - string_len;
    int left_pad = floor(diff_len / 2.0);
    int right_pad = ceil(diff_len / 2.0);

    *ptr_final_strings = strpad(*ptr_nizi, left_pad, right_pad);

    ptr_nizi++;
    ptr_final_strings++;
  }

  *ptr_final_strings = NULL;

  return final_strings;
}

//=============================================================================

#ifndef test

char *NIZI[] = {"Danes", "je", "kolokvij", "pri_P2!", NULL};

int main() {
  char **izhodni = naSredino(NIZI, 9);

  int stNizov = sizeof(NIZI) / sizeof(NIZI[0]) - 1;
  for (int i = 0; i < stNizov; i++) {
    printf("%d: \"%s\"\n", i, izhodni[i]);
    free(izhodni[i]);
  }
  printf("%s\n", (izhodni[stNizov] == NULL) ? ("NULL") : ("NAPAKA"));

  free(izhodni);
  return 0;
}

#endif
