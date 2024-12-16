#include <_stdlib.h>
#include <algorithm>
#include <iostream>
#include <bits/stdc++>

using namespace std;

int main() {
  int st_dni, st_sladic;

  cin >> st_sladic >> st_dni;

  vector<int> sladice;
  vector<int> sladice_summed;

  for (int i = 0; i < st_sladic; i++) {
    int sladica;
    cin >> sladica;
    sladice.push_back(sladica);
  }

  sort(sladice.begin(), sladice.end());

  int sum = 0;
  for (auto sladica : sladice) {
    sum += sladica;

    sladice_summed.push_back(sum);
  }

  for (int i = 0; i < st_dni; i++) {
    int dan;
    cin >> dan;

    auto num = upper_bound(sladice_summed.begin(), sladice_summed.end(), dan);

    if (i + 1 == st_dni) {
      cout << num - sladice_summed.begin();
    } else {
      cout << num - sladice_summed.begin() << " ";
    }
  }

  return 0;
}
