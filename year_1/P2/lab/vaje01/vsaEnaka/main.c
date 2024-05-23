#include <stdio.h>

int main(int argc, char **argv) {
  int n;
  int prev;

  scanf("%i", &n);
  scanf("%i", &prev);

  for (int i = 0; i < n - 1; i++) {
    int cur;
    scanf("%i", &cur);

    if (cur != prev) {
      printf("0\n");
      return 0;
    }
  }

  printf("1\n");
  return 0;

  return 0;
}
