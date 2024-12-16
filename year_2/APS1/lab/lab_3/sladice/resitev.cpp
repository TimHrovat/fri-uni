#include <_stdlib.h>
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main() {
  int n, m;
  cin >> n >> m;

  vector<int> x(n);
  vector<long> s(n);

  for (int i = 0; i < n; i++)
    cin >> x[i];

  sort(x.begin(), x.end());

  s[0] = x[0];

  for (int i = 1; i < n; i++) {
    s[i] = s[i - 1] + x[i];
  }

  for (int i = 0; i < m; i++) {
    int z;

    cin >> z;

    auto it = upper_bound(s.begin(), s.end(), z);

    cout << it - s.begin() << " ";
  }

  return 0;
}
