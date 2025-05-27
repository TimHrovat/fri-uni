#include <stdio.h>

int main(int argc, char **argv) {
  int n;
  int max;
  int sec_max;

  scanf("%i", &n);
  scanf("%i", &max);
  scanf("%i", &sec_max);

  if (max < sec_max) {
    int t = max;
    max = sec_max;
    sec_max = t;
  }

  for (int i = 0; i < n - 2; i++) {
    int current;

    scanf("%i", &current);

    if (current > max) {
        sec_max = max;
        max = current;
    } else if (current > sec_max) {
        sec_max = current;
    }
  }

  printf("%d\n", sec_max);

  return 0;
}
