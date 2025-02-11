#include <vector>
#include <iostream>

using namespace std;

int kviz(int k, vector<pair<int, int>> &tocke) {
  auto [a, b] = tocke[0];
  int suma = a;
  int maxb = b;

  int currmax = a + (k - 1) * b;

  for (int i = 1; i < k; i++) {
    tie(a, b) = tocke[i];

    if (b > maxb) {
      maxb = b;
    }
    suma += a;

    int kandidat = suma + (k - 1 - i) * maxb;
    if (kandidat > currmax) {
      currmax = kandidat;
    }
  }

  return currmax;
}

int main() {
  vector<pair<int, int>> tocke = {{2,3}, {4, 1}};
  int x = kviz(4, tocke);

  cout << x;
}
