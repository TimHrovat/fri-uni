#include <stdio.h>
#include <stdlib.h>

#define WIDTH 8
#define HEIGHT 8

typedef struct _Vozlisce {
  int data;
  struct _Vozlisce *next;
} Vozlisce;

int main() {
  int n;
  scanf("%d", &n);
  int matrix[HEIGHT][WIDTH] = {
      {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
  };

  for (int i = 0; i < n; i++) {
  }
}
