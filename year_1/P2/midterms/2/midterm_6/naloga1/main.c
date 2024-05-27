#include <limits.h>
#include <stdio.h>

#define MAX_N 1000000
#define MAX_ABS_VALUE 105

int main() {
  int n;
  scanf("%d", &n);

  int seq[MAX_N];
  int first_occurrence[2 * MAX_ABS_VALUE + 1];
  int last_occurrence[2 * MAX_ABS_VALUE + 1];

  // Initializing the occurrence arrays
  for (int i = 0; i < 2 * MAX_ABS_VALUE + 1; ++i) {
    first_occurrence[i] = -1;
    last_occurrence[i] = -1;
  }

  // Reading the sequence and updating occurrences
  for (int i = 0; i < n; ++i) {
    scanf("%d", &seq[i]);
    int idx = seq[i] + MAX_ABS_VALUE;
    if (first_occurrence[idx] == -1) {
      first_occurrence[idx] = i;
    }
    last_occurrence[idx] = i;
  }

  int max_distance = 0;
  for (int i = 0; i < 2 * MAX_ABS_VALUE + 1; ++i) {
    if (first_occurrence[i] != -1 &&
        last_occurrence[i] != first_occurrence[i]) {
      int distance = last_occurrence[i] - first_occurrence[i];
      if (distance > max_distance) {
        max_distance = distance;
      }
    }
  }

  printf("%d\n", max_distance);

  if (max_distance > 0) {
    for (int i = 0; i < 2 * MAX_ABS_VALUE + 1; ++i) {
      if (first_occurrence[i] != -1 &&
          last_occurrence[i] - first_occurrence[i] == max_distance) {
        printf("%d\n", i - MAX_ABS_VALUE);
      }
    }
  }

  return 0;
}
