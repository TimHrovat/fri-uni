
/*
Ro"cno poganjanje testnih programov (npr. test01.c):

gcc -Dtest test01.c naloga2.c
./a.out

Samodejno testiranje:

export name=naloga2
make test

Testni primeri:

test01: primer iz besedila
test02..test08: ro"cno izdelani kratki testi
test09..test10: samodejno izdelani, zadnji element prvega seznama < prvi element
drugega seznama test11..test16: samodejno izdelani, splo"sni

.c: testni program (prebere testne podatke, pokli"ce va"so funkcijo in izpi"se
rezultat) .dat: testni podatki .out: pri"cakovani izhod .res: izhod va"sega
programa (pri poganjanju z make)
*/

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "naloga2.h"

// Lahko dodate "se kak #include, pomo"zno funkcijo ipd.

//============================================================================

Vozlisce *zlij(Vozlisce *a, Vozlisce *b) {
  // Base cases
  if (a == NULL)
    return b;
  if (b == NULL)
    return a;

  // Recursive merge based on comparison
  if (a->podatek <= b->podatek) {
    a->naslednje = zlij(a->naslednje, b);
    return a;
  } else {
    b->naslednje = zlij(a, b->naslednje);
    return b;
  }
}

//============================================================================

// Vrstici z #ifndef in #endif pustite pri miru!

#ifndef test

Vozlisce *ustvari(int *t, int n, Vozlisce **podatek2vozlisce) {
  Vozlisce *zacetek = NULL;
  Vozlisce *konec = NULL;
  for (int i = 0; i < n; i++) {
    Vozlisce *novo = calloc(1, sizeof(Vozlisce));
    podatek2vozlisce[novo->podatek = t[i]] = novo;
    if (konec == NULL) {
      zacetek = konec = novo;
    } else {
      konec->naslednje = novo;
      konec = novo;
    }
  }
  return zacetek;
}

void pobrisi(Vozlisce *zacetek) {
  Vozlisce *p = zacetek;
  while (p != NULL) {
    Vozlisce *q = p->naslednje;
    free(p);
    p = q;
  }
}

void izpisi(Vozlisce *zacetek, Vozlisce **podatek2vozlisce) {
  printf("[");
  bool prvic = true;
  Vozlisce *p = zacetek;
  while (p != NULL) {
    if (prvic) {
      prvic = false;
    } else {
      printf(", ");
    }
    printf("%d", p->podatek);
    p = p->naslednje;
  }
  printf("]\n");

  p = zacetek;
  while (p != NULL) {
    printf("%d", p == podatek2vozlisce[p->podatek]);
    p = p->naslednje;
  }
  printf("\n");
}

int main() {
  FILE *f = fopen("test04.dat", "r");
  int da = 0;
  fscanf(f, "%d", &da);
  int *ta = malloc(da * sizeof(int));
  for (int i = 0; i < da; i++) {
    fscanf(f, "%d", &ta[i]);
  }
  int db = 0;
  fscanf(f, "%d", &db);
  int *tb = malloc(db * sizeof(int));
  for (int i = 0; i < db; i++) {
    fscanf(f, "%d", &tb[i]);
  }
  fclose(f);

  Vozlisce **podatek2vozlisce = malloc(200001 * sizeof(Vozlisce *));

  Vozlisce *a = ustvari(ta, da, podatek2vozlisce);
  Vozlisce *b = ustvari(tb, db, podatek2vozlisce);

  Vozlisce *c = zlij(a, b);

  izpisi(c, podatek2vozlisce);
  pobrisi(c);

  free(ta);
  free(tb);
  free(podatek2vozlisce);

  return 0;
}

#endif
