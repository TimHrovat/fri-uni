#include <_stdlib.h>
#include <iostream>

using namespace std;

typedef long long int64;
typedef vector<int> VectorInt;

pair<VectorInt, int64> merge(VectorInt &levo, VectorInt &desno) {
  int i = 0, j = 0;
  VectorInt c;

  int64 inv = 0;

  while (i < levo.size() || j < desno.size()) {
    if (i < levo.size() && j < desno.size()) {
      if (levo[i] <= desno[j]) {
        c.push_back(levo[i++]);
        inv += j;
      } else {
        c.push_back(desno[j++]);
      }
    } else if (i < levo.size()) {
      c.push_back(levo[i++]);
      inv += j;
    } else {
      c.push_back(desno[j++]);
    }
  }

  return {c, inv};
}

pair<VectorInt, int64> msort(VectorInt &sez) {
  int n = sez.size();

  if (n <= 1)
    return {sez, 0};

  VectorInt levo(sez.begin(), sez.begin() + n / 2);
  VectorInt desno(sez.begin() + n / 2, sez.end());

  auto [levo_urejen, li] = msort(levo);
  auto [desno_urejen, di] = msort(desno);

  auto [skupaj, inv] = merge(levo_urejen, desno_urejen);

  return {skupaj, li + di + inv};
}

int main() {
  int n;

  std::cin >> n;

  VectorInt v(n);

  for (int i = 0; i < n; i++)
    cin >> v[i];

  auto [s, inv] = msort(v);

  cout << inv << endl;

  return 0;
}
