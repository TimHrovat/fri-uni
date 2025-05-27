
#include "naloga2.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int izpisiB(B* b, char* cilj);
int izpisiC(C* c, char* cilj);
int digit_count(int num);

int digit_count(int num) {
    int count = num >= 0 ? 1 : 2;
    num = abs(num);

    while (num > 10) {
        count++;
        num = num / 10;
    }

    return count;
}

int izpisiA(A* a, char* cilj) {
    if (a == NULL) {
        strcat(cilj, "NULL");
        return 4;
    }

    int len = digit_count(a->p) + 4;
    char str[len + 3];

    sprintf(str, "{%d, ", a->p);
    strcat(cilj, str);
    len += izpisiB(a->b, cilj);
    strcat(cilj, "}");

    return len;
}

int izpisiB(B* b, char* cilj) {
    if (b == NULL) {
        strcat(cilj, "NULL");
        return 4;
    }

    int len = strlen(b->q) + 4;

    strcat(cilj, "{");
    strcat(cilj, b->q);
    strcat(cilj, ", ");
    len += izpisiC(b->c, cilj);
    strcat(cilj, "}");

    return len;
}

int izpisiC(C* c, char* cilj) {
    if (c == NULL) {
        strcat(cilj, "NULL");
        return 4;
    }

    int len = (c->r ? 4 : 5) + 6;

    strcat(cilj, "{");
    strcat(cilj, c->r ? "true" : "false");
    strcat(cilj, ", ");
    len += izpisiA(c->a, cilj);
    strcat(cilj, ", ");
    len += izpisiB(c->b, cilj);
    strcat(cilj, "}");

    return len;
}

#ifndef test

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

#endif
