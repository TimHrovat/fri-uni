
/*

Prevajanje in poganjanje:

gcc -o test01 test01.c inverz.c -lm
./test01

*/

#include "inverz.h"

#include <stdbool.h>

long inverz(long x, long a, long b) {
    long zac = a;
    long kon = b;
    long curr = kon / 2;

    do {
        if (x == f(curr)) {
            return curr;
        }

        if (x > f(curr)) {
            zac = curr + 1;
        } else {
            kon = curr - 1;
        }

        curr = zac + (kon - zac) / 2;
    } while (zac != kon || (curr == kon && curr == zac));

    return -1;
}

// Ta datoteka NE SME vsebovati funkcij main in f!
// Funkciji main in f sta definirani v datoteki test01.c.
