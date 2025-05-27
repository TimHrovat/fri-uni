
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "naloga.h"

int vsotaI(Vozlisce *zacetek) {
  Vozlisce *_zacetek = zacetek;
  int sum = 0;

  while (_zacetek != NULL) {
    sum += _zacetek->podatek;
    _zacetek++;
  }
  // popravite / dopolnite ...
  return sum;
}

int vsotaR(Vozlisce *zacetek) {
  if (zacetek == NULL) {
    return 0;
  }
  // popravite / dopolnite ...
  return zacetek->podatek + vsotaR(zacetek->naslednje);
}

Vozlisce *vstaviUrejenoI(Vozlisce *zacetek, int element) {
  // popravite / dopolnite ...
  return NULL;
}

Vozlisce *vstaviUrejenoR(Vozlisce *zacetek, int element) {
  if (zacetek == NULL || element < zacetek->podatek) {
    Vozlisce *novoVozlisce = (Vozlisce *)malloc(sizeof(Vozlisce));
    novoVozlisce->podatek = element;
    novoVozlisce->naslednje = zacetek;
    return novoVozlisce;
  }

  zacetek->naslednje = vstaviUrejenoR(zacetek->naslednje, element);
  return zacetek;
}

#ifndef test

int main() {
  // koda za ro"cno testiranje (po "zelji)

  return 0;
}

#endif
