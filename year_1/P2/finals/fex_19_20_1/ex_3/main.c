#include <stdio.h>

void print_subset(int set[], int size);
void _print_set(int set[], int size, int subset[], int subset_size, int index);
void print_set(int set[], int size);

int main(int argc, char **argv) {
  FILE *file = fopen(argv[1], "r");

  // read set size
  int n;
  fscanf(file, "%d", &n);

  // read set
  int set[n];
  for (int i = 0; i < n; i++) {
    fscanf(file, "%d", &set[i]);
  }

  print_set(set, n);

  fclose(file);

  return 0;
}

void print_subset(int set[], int size) {
  printf("{");
  for (int i = 0; i < size; i++) {
    printf("%d", set[i]);

    if (i + 1 != size) {
      printf(", ");
    }
  }
  printf("}\n");
}

void _print_set(int set[], int size, int subset[], int subset_size, int index) {
  if (size == index) {
    print_subset(subset, subset_size);
    return;
  }

  _print_set(set, size, subset, subset_size, index + 1);

  subset[subset_size] = set[index];
  _print_set(set, size, subset, subset_size + 1, index + 1);
}

void print_set(int set[], int size) {
  int subset[size];
  _print_set(set, size, subset, 0, 0);
}
