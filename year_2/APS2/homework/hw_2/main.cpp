#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

int comparisons = 0;
void countComparison() { comparisons++; }
void resetComparisonCounter() { comparisons = 0; }

vector<int> generateSortedTable(int size);
vector<int> generateRandomTable(int size);
int pivotFirst(vector<int> &table, int left, int right);
int pivotRandom(vector<int> &table, int left, int right);
int pivotRandomThree(vector<int> &table, int left, int right);
int pivotMedianOfMedians(vector<int> &table, int left, int right);
int quickSelect(vector<int> &table, int left, int right, int k,
                int (*pivotFunc)(vector<int> &, int, int));

int main() {
  srand(time(0));
  vector<pair<int, int>> TESTS = {
      {100, 10}, {500, 50}, {1000, 100}, {5000, 500}};
  vector<pair<string, int (*)(vector<int> &, int, int)>> PIVOT_GENERATORS = {
      {"prvi", pivotFirst},
      {"nakljucni", pivotRandom},
      {"mediana treh", pivotRandomThree},
      {"mediana median", pivotMedianOfMedians}};

  cout << "1. tabela povprecnega st. primerjav pri nakljucnih tabelah" << endl;
  cout << setw(20) << "";
  for (auto &test : TESTS) {
    cout << setw(14) << left << test.first;
  }
  cout << endl;
  for (auto &pivotGenerator : PIVOT_GENERATORS) {
    cout << setw(20) << left << pivotGenerator.first;
    for (auto &test : TESTS) {
      for (int i = 0; i < test.second; i++) {
        auto table = generateRandomTable(test.first);
        int k = rand() % (table.size());
        quickSelect(table, 0, table.size() - 1, k, pivotGenerator.second);
      }
      cout << setw(14) << comparisons / test.second;
      resetComparisonCounter();
    }
    cout << endl;
  }

  cout << endl << endl;

  cout << "2. tabela povprecnega st. primerjav pri narascajoce urejenih tabelah" << endl;
  cout << setw(20) << "";
  for (auto &test : TESTS) {
    cout << setw(14) << left << test.first;
  }
  cout << endl;
  for (auto &pivotGenerator : PIVOT_GENERATORS) {
    cout << setw(20) << left << pivotGenerator.first;
    for (auto &test : TESTS) {
      for (int i = 0; i < test.second; i++) {
        auto table = generateSortedTable(test.first);
        int k = rand() % (table.size());
        quickSelect(table, 0, table.size() - 1, k, pivotGenerator.second);
      }
      cout << setw(14) << comparisons / test.second;
      resetComparisonCounter();
    }
    cout << endl;
  }

  return 0;
}

int pivotFirst(vector<int> &table, int left, int right) { return left; }
int pivotRandom(vector<int> &table, int left, int right) {
  return left + rand() % (right - left + 1);
}
int pivotRandomThree(vector<int> &table, int left, int right) {
  int a = pivotRandom(table, left, right);
  int b = pivotRandom(table, left, right);
  int c = pivotRandom(table, left, right);
  if (table[a] > table[b])
    swap(a, b);
  if (table[b] > table[c])
    swap(b, c);
  if (table[a] > table[b])
    swap(a, b);
  return b;
}
int pivotMedianOfMedians(vector<int> &table, int left, int right) {
  int n = right - left + 1;
  if (n <= 5) {
    sort(table.begin() + left, table.begin() + right + 1);
    return (left + right) / 2;
  }

  for (int i = left; i <= right; i += 5) {
    int groupEnd = min(i + 4, right);
    sort(table.begin() + i, table.begin() + groupEnd + 1);
    int medianIndex = (i + groupEnd) / 2;
    swap(table[medianIndex], table[left + (i - left) / 5]);
  }

  return pivotMedianOfMedians(table, left, left + (right - left) / 5);
}

int quickSelect(vector<int> &table, int left, int right, int k,
                int (*pivotFunc)(vector<int> &, int, int)) {
  if (left == right)
    return table[left];

  int pivotIndex = pivotFunc(table, left, right);
  swap(table[pivotIndex], table[right]);
  int pivotValue = table[right];

  int i = left;
  for (int j = left; j < right; ++j) {
    countComparison();
    if (table[j] < pivotValue) {
      swap(table[i], table[j]);
      i++;
    }
  }
  swap(table[i], table[right]);

  if (k == i)
    return table[i];
  else if (k < i)
    return quickSelect(table, left, i - 1, k, pivotFunc);
  else
    return quickSelect(table, i + 1, right, k, pivotFunc);
}

vector<int> generateSortedTable(int size) {
  vector<int> table(size);
  for (int i = 0; i < size; ++i) {
    table[i] = i + 1;
  }
  return table;
}

vector<int> generateRandomTable(int size) {
  vector<int> table = generateSortedTable(size);

  random_device rd;
  mt19937 g(rd());
  shuffle(table.begin(), table.end(), g);

  return table;
}
