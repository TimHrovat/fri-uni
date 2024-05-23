
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

int dist(int point_a[], int point_b[], int num_of_coords);

int main() {
  int num_of_coords;
  scanf("%d", &num_of_coords);
  int point_a[num_of_coords];
  int point_b[num_of_coords];

  for (int i = 0; i < num_of_coords; i++) {
    scanf("%d", &point_a[i]);
  }

  for (int i = 0; i < num_of_coords; i++) {
    scanf("%d", &point_b[i]);
  }

  int distance = dist(point_a, point_b, num_of_coords);

  printf("%d\n", distance);

  return 0;
}

int dist(int point_a[], int point_b[], int num_of_coords) {
  if (num_of_coords == 0) {
    return 0;
  }

  int distance = point_a[num_of_coords - 1] - point_b[num_of_coords - 1];

  return pow(distance, 2) + dist(point_a, point_b, num_of_coords - 1);
}
