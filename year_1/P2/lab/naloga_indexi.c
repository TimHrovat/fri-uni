#include <stdio.h>
#include <stdlib.h>

#define N 5

int **create_table(int n);

int main() {
  int **tabela = create_table(N);

  for (int i = 0; i < N; i++) {
    printf("%d: ", tabela[i][-1]);

    for (int j = 0; j < i; j++) {
      printf("%d, ", tabela[i][j]);
    }

    printf("\n");
  }

  printf("%s", tabela[N] == NULL ? "NULL" : "NOT NULL");

  return 0;
}

int **create_table(int n) {
  int **tabela = (int **)malloc(sizeof(int **) * (n + 1));
  int **ptr_tabela = tabela;

  for (int i = 0; i < n; i++) {
    tabela[i] = malloc(sizeof(int *) * (i + 2));

    tabela[i][0] = i;
    for (int j = 1; j < i + 1; j++) {
      tabela[i][j] = j - 1;
    }

    ptr_tabela++;
    tabela[i]++;
  }

  ptr_tabela[n] = NULL;

  return tabela;
}
