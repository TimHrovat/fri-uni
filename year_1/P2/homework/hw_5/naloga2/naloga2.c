
/*

Prevajanje in poganjanje skupaj z datoteko test01.c:

gcc -D=test test01.c naloga2.c
./a.out


*/

#include "naloga2.h"

#include <stdio.h>

void zamenjaj(int** a, int** b) {
    int* tmp = *a;
    *a = *b;
    *b = tmp;
}

void uredi(int** a, int** b, int** c) {
    if (**a >= **b) {
        zamenjaj(a, b);
    }

    if (**c < **a) {
        zamenjaj(a, c);
        zamenjaj(b, c);
    } else if (**c > **a && **c < **b) {
        zamenjaj(b, c);
    }
}

#ifndef test

int main() {
    int a = 70;
    int b = 90;
    int* pa = &a;
    int* pb = &b;

    zamenjaj(&pa, &pb);

    printf("%d, %d", *pa, *pb);

    return 0;
}

#endif
