
#include "naloga1.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void _zdesetkaj(Vozlisce** start, Vozlisce** last_match, Vozlisce* cur, int k, int i);

Vozlisce* zdesetkaj(Vozlisce* zacetek, int k) {
    Vozlisce* new_start = NULL;
    Vozlisce* last_match = NULL;

    _zdesetkaj(&new_start, &last_match, zacetek, k, 1);

    return new_start;
}

void _zdesetkaj(Vozlisce** start, Vozlisce** last_match, Vozlisce* cur, int k, int i) {
    if (cur == NULL) {
        return;
    }

    bool is_kth = i % k == 0;
    Vozlisce* next = cur->naslednje;

    if (is_kth && *start == NULL) {
        *start = cur;
        *last_match = cur;
    }

    if (is_kth) {
        (*last_match)->naslednje = cur;

        *last_match = (*last_match)->naslednje;

        (*last_match)->naslednje = NULL;
    } else {
        free(cur);
    }

    _zdesetkaj(start, last_match, next, k, ++i);
}

#ifndef test

int main() { return 0; }

#endif
