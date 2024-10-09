
/*
Ro"cno testiranje (npr. za primer test01.in):

gcc naloga3.c
./a.out < test01.in

Samodejno testiranje:

export name=naloga3
make test

Testni primeri:

test01: primer iz besedila
test02..test08: "se nekaj dodatnih testov

.in: testni vhod
.outA: pri"cakovani izhod (poljubna permutacija vrstic je tudi v redu)
.outB: pri"cakovani izhod za 0.3 to"cke (poljubna permutacija vrstic je tudi v
redu) .res: izhod va"sega programa (pri poganjanju z make)
*/

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Lahko dodate "se kak #include, pomo"zno funkcijo ipd.
void _f(char *str, int d, int target_len, char *buf, int len) {
  if (len == target_len) {
    buf[target_len] = '\0';
    printf("%s\n", buf);
    return;
  }

  for (int i = 0; i < d; i++) {
    int times_used = 0;
    for (int j = 0; j < len; j++) {
      if (buf[j] == str[i]) {
        times_used++;
      }
    }

    if (times_used < 3) {
      buf[len] = str[i];
      _f(str, d, target_len, buf, len + 1);
    }
  }
}

void f(char *str, int d, int target_len) {
  char *buf = (char *)calloc(100, sizeof(char));

  _f(str, d, target_len, buf, 0);
}

int main() {
  int d, target_len;
  char *str = (char *)calloc(50, sizeof(char));
  scanf("%d %s %d", &d, str, &target_len);

  f(str, d, target_len);

  return 0;
}
