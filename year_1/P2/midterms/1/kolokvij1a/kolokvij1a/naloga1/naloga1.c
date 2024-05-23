
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// po potrebi dopolnite ...

int main() {
  int len;
  scanf("%d", &len);
  int *arr = (int *)malloc(sizeof(int *) * len);
  int *ptr_arr = arr;

  long sum = 0;
  for (int i = 0; i < len; i++) {
    scanf("%d", ptr_arr);
    sum += *ptr_arr;

    ptr_arr++;
  }

  bool is_palindrom = true;
  ptr_arr = arr;
  for (int i = 0; i < len / 2; i++) {
    int a = ptr_arr[i];
    int b = ptr_arr[len - (i + 1)];

    if (a != b) {
        is_palindrom = false;
        break;
    }
  }

  printf("%s\n%ld\n", is_palindrom ? "DA" : "NE", sum);

  free(arr);

  return 0;
}
