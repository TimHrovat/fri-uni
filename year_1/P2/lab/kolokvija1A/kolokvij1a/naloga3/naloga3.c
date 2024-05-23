
/*
 * Zagon testne skripte ("sele potem, ko ste prepri"cani, da program deluje!):
 *
 * export name=naloga3
 * make test
 */

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int *count_rec(int num, int min_divisor);

int main() {
  int num, min_divisor;
  scanf("%d %d", &num, &min_divisor);

  for (int i = min_divisor; i <= num; i++) {
    for (int j = min_divisor; j <= num; j++) {

    }
  }

  return 0;
}

