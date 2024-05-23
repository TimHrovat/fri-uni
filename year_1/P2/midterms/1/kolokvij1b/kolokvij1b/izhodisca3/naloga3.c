
/*
 * Zagon testne skripte ("sele potem, ko ste prepri"cani, da program deluje!):
 *
 * export name=naloga3
 * make test
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// po potrebi dopolnite ...

bool is_candidate(int num, int n, int m);
long combinations(int a, int b);

int main() {
  int n, m;
  scanf("%d", &n);
  scanf("%d", &m);

  long count = 0;
  for (int i = 3; i <= n; i++) {
    if (!is_candidate(i, n, m)) {
      continue;
    }

    count += combinations(i, n / i);
    printf("%d\n", i);
  }

  return 0;
}

bool is_candidate(int num, int n, int m) {
  if (num > m) {
    return false;
  }

  if (num == n / 2) {
    return false;
  }

  return n % num == 0;
}

long combinations(int a, int b) {
  long count = 0;

  for (int i = ceil(a / 2.0), i > 1, i--) {
  }

  return count;
}
