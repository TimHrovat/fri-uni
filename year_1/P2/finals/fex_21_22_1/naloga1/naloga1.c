
/*
Ro"cno poganjanje testnih programov (npr. test01.c):

gcc -Dtest test01.c naloga1.c
./a.out

Samodejno testiranje:

export name=naloga1
make test

Testni primeri:

test01..test03: ro"cno izdelani kratki testi
test04: samodejno izdelani, predpona = +386, dol"zina tel. "st. = 8
test05: samodejno izdelani, predpona = +386
test06..test10: samodejno izdelani, splo"sni

.c: testni program (prebere testne podatke, pokli"ce va"so funkcijo in izpi"se
rezultat) .dat: testni podatki .out: pri"cakovani izhod .res: izhod va"sega
programa (pri poganjanju z make)
*/

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "naloga1.h"

// Lahko dodate "se kak #include, pomo"zno funkcijo ipd.

//============================================================================

void vstaviPredpono(char *predpona, Oseba **osebe, int stOseb) {
  if (stOseb == 0) {
    return;
  }

  Oseba *oseba = *osebe;

  char *buf = calloc(MAXFON, sizeof(char));
  strcat(buf, predpona);
  strcat(buf, oseba->telefon);
  strncpy(oseba->telefon, buf, MAXFON);
  free(buf);

  vstaviPredpono(predpona, osebe + 1, stOseb - 1);
}

//============================================================================

// Vrstici z #ifndef in #endif pustite pri miru!

#ifndef test

int main(int argc, char **argv) {
  FILE *f = fopen("test01.dat", "r");
  int stOseb = 0;
  fscanf(f, "%d", &stOseb);

  char ime[100] = {'\0'};
  char predpona[100] = {'\0'};

  Oseba **osebe = malloc(stOseb * sizeof(Oseba *));
  for (int i = 0; i < stOseb; i++) {
    Oseba *oseba = calloc(1, sizeof(Oseba));
    fscanf(f, "%s%s", ime, oseba->telefon);
    oseba->ime = calloc(strlen(ime) + 1, sizeof(char));
    strcpy(oseba->ime, ime);
    osebe[i] = oseba;
  }
  fscanf(f, "%s", predpona);
  fclose(f);

  vstaviPredpono(predpona, osebe, stOseb);

  for (int i = 0; i < stOseb; i++) {
    printf("%s / %s\n", osebe[i]->ime, osebe[i]->telefon);
  }

  for (int i = 0; i < stOseb; i++) {
    free(osebe[i]->ime);
    free(osebe[i]);
  }
  free(osebe);
  return 0;
}

#endif
