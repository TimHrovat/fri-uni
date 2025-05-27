#include <stdio.h>

int countWays(int arr[], int n, int k, int v, int index, int currentCount,
              int currentSum) {
  // Če smo izbrali k elementov in je njihova vsota v, vrnemo 1 (najdena
  // rešitev)
  if (currentCount == k) {
    return (currentSum == v) ? 1 : 0;
  }

  // Če smo presegli velikost tabele ali če ostali elementi ne zadostujejo,
  // vrnemo 0 (brez rešitve)
  if (index == n || currentCount > k) {
    return 0;
  }

  // Število načinov, če vključimo trenutni element
  int include = countWays(arr, n, k, v, index + 1, currentCount + 1,
                          currentSum + arr[index]);

  // Število načinov, če izpustimo trenutni element
  int exclude = countWays(arr, n, k, v, index + 1, currentCount, currentSum);

  // Skupno število načinov
  return include + exclude;
}

int main() {
  int n, k, v;

  // Branje vhodnih podatkov
  scanf("%d %d %d", &n, &k, &v);

  int arr[n];
  for (int i = 0; i < n; i++) {
    scanf("%d", &arr[i]);
  }

  // Iskanje vseh načinov
  int result = countWays(arr, n, k, v, 0, 0, 0);

  // Izpis rezultata
  printf("%d\n", result);

  return 0;
}
