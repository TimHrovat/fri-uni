#include <stdio.h>
#include <stdlib.h>

typedef struct _Vozlisce {
  struct _Vozlisce *naslednje;
} Vozlisce;

int cycle_len(Vozlisce *p);
int _cycle_len(Vozlisce *p, Vozlisce *first);

int main() {
  Vozlisce *p = (Vozlisce *)malloc(sizeof(Vozlisce));
  Vozlisce *first = p;

  for (int i = 0; i < 10; i++) {
    p->naslednje = (Vozlisce *)malloc(sizeof(Vozlisce));
    p = p->naslednje;
  }

  p->naslednje = first;

  printf("%d", cycle_len(p));
}

int cycle_len(Vozlisce *p) { return _cycle_len(p, p); }

int _cycle_len(Vozlisce *p, Vozlisce *first) {
  if (p->naslednje == first) {
    return 1;
  }

  return 1 + _cycle_len(p->naslednje, first);
}
