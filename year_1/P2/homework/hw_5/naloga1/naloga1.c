/*
Prevajanje in poganjanje skupaj z datoteko test01.c:

gcc -D=test test01.c naloga1.c
./a.out
*/

#include "naloga1.h"

int* poisci(int* t, int* dolzina, int** konec) {
    // popravite / dopolnite ...
    int* pointer_t = t;

    while (*(pointer_t + 1) != 0) {
        pointer_t++;
    }

    *konec = pointer_t;

    pointer_t = t;

    while (*(pointer_t - 1) != 0) {
        pointer_t--;
    }

    int* zacetek = pointer_t;

    int count = 0;
    while (*pointer_t != 0) {
        count++;
        pointer_t++;
    }

    *dolzina = count;

    return zacetek;
}

#ifndef test

int main() { return 0; }

#endif
