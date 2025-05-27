
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "naloga2.h"

int main() {
    A* a = malloc(sizeof(A));
    B* b = malloc(sizeof(B));
    C* c = malloc(sizeof(C));
    A* a2 = malloc(sizeof(A));
    B* b2 = malloc(sizeof(B));
    C* c2 = malloc(sizeof(C));

    a->p = 42;
    a->b = b;

    b->q = "dober";
    b->c = c;

    c->r = true;
    c->a = a2;
    c->b = b2;

    a2->p = -15;
    a2->b = NULL;

    b2->q = "dan";
    b2->c = c2;

    c2->r = false;
    c2->a = NULL;
    c2->b = NULL;

    char* cilj = malloc(10010 * sizeof(char));
    int stZnakov = izpisiA(a, cilj);
    printf("%s\n", cilj);
    printf("%d\n", stZnakov);

    free(a);
    free(b);
    free(c);
    free(a2);
    free(b2);
    free(c2);

    return 0;
}
