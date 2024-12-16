#include <iostream>

using namespace std;

#define N 1'000'000

int n;
int d[N + 1];
vector<int> sub[N + 1];
int total[N + 1];

void solve(int x, int sum = 0) {
  total[x] = sum + d[x];

  for (auto child : sub[x]) {
    solve(child, total[x]);
  }
}

int main() {
  cin >> n;

  // gradimo drevo
  for (int i = 1; i <= n; i++) {
    int p;
    cin >> p;
    sub[p].push_back(i); // i je podrejeni p[i]
  }

  for (int i = 1; i <= n; i++) {
    cin >> d[i];
  }

  for (auto x : sub[0]) {
    solve(x);
  }

  for (int i = 1; i <= n; i++) {
    cout << total[i] << " ";
  }

  return 0;
}
