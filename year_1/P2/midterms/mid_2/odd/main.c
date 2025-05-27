#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
  int value;
  struct Node *next;
} Node;

Node *prepisi(int *t);
int odd(Node *node);

int main() {
  int tab[] = {1, 2, 3, -4, 5, 6, 7, 8, 9, -10, 11, 0};

  Node *vozlisce = prepisi(tab);

  printf("%d", odd(vozlisce));

  while (vozlisce != NULL) {
    Node *naslednje = vozlisce->next;

    free(vozlisce);

    vozlisce = naslednje;
  }

  return 0;
}

Node *prepisi(int *t) {
  if (*t == 0) {
    return NULL;
  }

  Node *vozlisce = malloc(sizeof(Node));

  vozlisce->value = *t;
  vozlisce->next = prepisi(t + 1);

  return vozlisce;
}

int odd(Node *node) {
  if (node == NULL) {
    return 0;
  }

  int o = odd(node->next);

  return o == 1 ? 0 : 1;
}
