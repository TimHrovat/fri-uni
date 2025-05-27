
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "naloga.h"

int steviloZnakov(char *niz, char znak) {
  char *p = niz;
  char chr;
  int count = 0;

  if (niz == NULL)
    return 0;

  while (*p != '\0') {
    if (*p == znak)
      count++;

    p++;
  }

  return count;
}

char *kopirajDoZnaka(char *niz, char znak) {
  // char *tmp_niz = niz;
  //
  // while (*tmp_niz != '\0' && *tmp_niz != znak) {
  //   tmp_niz++;
  // }
  //
  // int len = tmp_niz - niz;
  //
  // char *final_niz = (char *)malloc((len + 1) * sizeof(char));
  //
  // final_niz[len] = '\0';
  //
  // char *p = niz;
  // char *q = final_niz;
  // for (int i = 0; i < len; i++) {
  //   *q++ = *p++;
  // }
  // *q = '\0';
  //
  // return final_niz;

  // rešitev s funkcijami
  int lenNiza = strlen(niz);
  char *pZnak = strchr(niz, znak);
  int lenPodniza = (pZnak == NULL) ? lenNiza : pZnak - niz;
  char *podniz = (char *)malloc((lenPodniza + 1) * sizeof(char));
  strncpy(podniz, niz, lenPodniza);
  podniz[lenPodniza] = '\0';
  return podniz;
}

char **razcleni(char *niz, char locilo, int *stOdsekov) {
  *stOdsekov = steviloZnakov(niz, locilo) + 1;
  char **odseki = (char **)malloc(*stOdsekov * sizeof(char *));
  char *tmp_niz = niz;

  for (int i = 0; i < *stOdsekov; i++) {
    odseki[i] = kopirajDoZnaka(tmp_niz, locilo);

    tmp_niz += strlen(odseki[i]) + 1;
  }

  return odseki;
}

#ifndef test

int main() {
  int n = 0;
  char **odseki = razcleni("svoje_delo_oddajte_najkasneje_do_nedelje", 'd', &n);

  for (int i = 0; i < n; i++) {
    printf("%s\n", odseki[i]);
  }

  return 0;
}

#endif
