#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
  int value;
  struct Node *next;
} Node;

Node *prepisi(int *t);
Node *minus1(Node *node);

int main() {
  int tab[] = {1, 2, 3, -4, 5, 6, 7, 8, 9, -10, 0};

  Node *vozlisce = prepisi(tab);

  printf("%d", minus1(vozlisce)->value);

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

Node *minus1(Node *node) {
  if (node == NULL || node->next == NULL || node->next->next == NULL) {
    return NULL;
  }

  Node *m_node = minus1(node->next);

  if (m_node != NULL) {
    return m_node;
  }

  if (node->next->value + node->next->next->value == -1) {
    return node;
  }

  return NULL;
}
