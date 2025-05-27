#include <stdio.h>
#include <stdlib.h>

typedef struct _Vozlisce {
  int podatek;
  struct _Vozlisce *naslednji;
} Vozlisce;

Vozlisce *prepisi(int *t);

int main() {
  int tab[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0};

  Vozlisce *vozlisce = prepisi(tab);

  while (vozlisce != NULL) {
    Vozlisce *naslednji = vozlisce->naslednji;

    printf("%d\n", vozlisce->podatek);
    free(vozlisce);

    vozlisce = naslednji;
  }

  return 0;
}

Vozlisce *prepisi(int *t) {
  if (*t == 0) {
    return NULL;
  }

  Vozlisce *vozlisce = malloc(sizeof(Vozlisce));

  vozlisce->podatek = *t;
  vozlisce->naslednji = prepisi(t + 1);

  return vozlisce;
}
