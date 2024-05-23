
#include "naloga2.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

int **ap2pp(int (*kazalec)[N], int izvornoStVrstic, int ciljnoStVrstic) {
    int **matrix = (int **)malloc(sizeof(int **) * ciljnoStVrstic);
    int **ptr_matrix = matrix;
    int *matrix_row;

    int count = 0;
    for (int i = 0; i < izvornoStVrstic; i++) {
        for (int j = 0; j < ciljnoStVrstic; j++) {
            if (count % izvornoStVrstic == 0) {
                *ptr_matrix = (int *)malloc(sizeof(int *) * (izvornoStVrstic + 1));
                matrix_row = *ptr_matrix;
                *(matrix_row + izvornoStVrstic) = 0;
                ptr_matrix++;
            }
            count++;

            *matrix_row = kazalec[i][j];

            matrix_row++;
        }
    }

    return matrix;
}

int (*pp2ap(int **kazalec, int izvornoStVrstic, int *ciljnoStVrstic))[N] {
    int **ptr_kazalec = kazalec;

    int count = 0;
    for (int i = 0; i < izvornoStVrstic; ++i) {
        int *row_kazalec = *ptr_kazalec;

        while (*row_kazalec != 0) {
            count++;
            row_kazalec++;
        }

        ptr_kazalec++;
    }

    *ciljnoStVrstic = (int)ceil((double)count / N);

    int(*matrix)[N] = malloc(sizeof(int) * (*ciljnoStVrstic) * N);
    int(*ptr_matrix)[N] = matrix;
    ptr_kazalec = kazalec;

    int *row_kazalec = *ptr_kazalec;
    for (int i = 0; i < *ciljnoStVrstic; ++i) {
        for (int j = 0; j < N; ++j) {
            if (*row_kazalec == 0 && (i * N + j + 1) <= count) {
                ptr_kazalec++;
                row_kazalec = *ptr_kazalec;
            }

            (*ptr_matrix)[j] = (i * N + j + 1) > count ? 0 : *row_kazalec;  // TODO

            row_kazalec++;
        }
        ptr_matrix++;
    }

    return matrix;
}

#ifndef test

int main() {
    int TABELA[][N] = {{5, 7, 12, 9, 8}, {1, 15, 3, 6, 14}, {11, 10, 2, 13, 4}};
    int izvornoStVrstic = sizeof(TABELA) / sizeof(TABELA[0]);
    int ciljnoStVrstic = 5;
    int **rezultat = ap2pp(TABELA, izvornoStVrstic, ciljnoStVrstic);
    // koda za ro"cno testiranje (po "zelji)
    return 0;
}

#endif
