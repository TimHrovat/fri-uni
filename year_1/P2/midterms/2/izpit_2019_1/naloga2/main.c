#include <stdio.h>
#include <stdlib.h>

typedef struct Vozlisce Vozlisce;
struct Vozlisce {
  int podatek;
  Vozlisce *naslednje;
};

Vozlisce *obrni(Vozlisce *zacetek);
Vozlisce *prepisi(int *t);

int main() {
  int tab[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0};

  Vozlisce *vozlisce = prepisi(tab);
  vozlisce = obrni(vozlisce);

  while (vozlisce != NULL) {
    Vozlisce *naslednje = vozlisce->naslednje;

    printf("%d\n", vozlisce->podatek);
    free(vozlisce);

    vozlisce = naslednje;
  }

  return 0;
}

Vozlisce *prepisi(int *t) {
  if (*t == 0) {
    return NULL;
  }

  Vozlisce *vozlisce = malloc(sizeof(Vozlisce));

  vozlisce->podatek = *t;
  vozlisce->naslednje = prepisi(t + 1);

  return vozlisce;
}

Vozlisce *obrni(Vozlisce *zacetek) {
  if (zacetek == NULL || zacetek->naslednje == NULL) {
    return zacetek;
  }

  Vozlisce *naslednje = obrni(zacetek->naslednje);

  zacetek->naslednje->naslednje = zacetek;
  zacetek->naslednje = NULL;

  return naslednje;
}
