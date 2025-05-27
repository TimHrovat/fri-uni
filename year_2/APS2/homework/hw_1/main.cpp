#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <vector>

#define TRIAL_REPS 1000

#define START_N 1000
#define END_N 100000
#define STEP 1000

using namespace std;

vector<int> generateTable(int n);
int findLinear(const vector<int> &a, int v);
int findBinary(const vector<int> &a, int l, int r, int v);
long timeLinear(int n);
long timeBinary(int n);

int main() {
  srand(time(nullptr));

  cout << setw(10) << "n" << " | ";
  cout << setw(14) << "linearno" << " | ";
  cout << setw(14) << "dvojisko" << " |" << endl;

  cout << "-----------+----------------+----------------+" << endl;

  for (int i = START_N; i <= END_N; i += STEP) {
    long linearTime = timeLinear(i);
    long binaryTime = timeBinary(i);

    cout << setw(10) << i << " | ";
    cout << setw(14) << linearTime << " | ";
    cout << setw(14) << binaryTime << " |" << endl;
  }

  return 0;
}

vector<int> generateTable(int n) {
  vector<int> vec;

  for (int i = 1; i <= n; i++) {
    vec.push_back(i);
  }

  return vec;
}

int findLinear(const vector<int> &a, int v) {
  int i = 0;

  for (auto &el : a) {
    if (el == v)
      return i;
    i++;
  }

  return -1;
}

int findBinary(const vector<int> &a, int l, int r, int v) {
  if (r >= l) {
    int mid = l + (r - l) / 2;

    if (a[mid] == v)
      return mid;

    if (a[mid] > v)
      return findBinary(a, l, mid - 1, v);

    return findBinary(a, mid + 1, r, v);
  }

  return -1;
}

long timeLinear(int n) {
  auto begin = chrono::high_resolution_clock::now();
  auto table = generateTable(n);

  for (int i = 0; i < TRIAL_REPS; i++) {
    int random_num = rand() % n + 1;

    findLinear(table, random_num);
  }

  auto end = chrono::high_resolution_clock::now();

  return chrono::duration_cast<chrono::nanoseconds>(end - begin).count() /
         TRIAL_REPS;
}

long timeBinary(int n) {
  auto begin = chrono::high_resolution_clock::now();
  auto table = generateTable(n);

  for (int i = 0; i < TRIAL_REPS; i++) {
    int random_num = rand() % n + 1;

    findBinary(table, 0, table.size() - 1, random_num);
  }

  auto end = chrono::high_resolution_clock::now();

  return chrono::duration_cast<chrono::nanoseconds>(end - begin).count() /
         TRIAL_REPS;
}
