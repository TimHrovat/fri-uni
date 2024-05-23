
/*

Prevajanje in poganjanje:

gcc -o test01 test01.c tranzitivnost.c -lm
./test01

*/

#include "tranzitivnost.h"

#include <stdbool.h>
#include <stdio.h>

int tranzitivnost(int a, int b) {
    // popravite / dopolnite ...
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < a; j++) {
            for (int i = 0; i < a; i++) {
            }
        }
    }

    return -1;
}

// Ta datoteka NE SME vsebovati funkcij main in f!
// Funkciji main in f sta definirani v datoteki test01.c.
