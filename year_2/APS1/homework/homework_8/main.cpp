#include <iostream>
#include <queue>
#include <vector>

using namespace std;

// bfs algo
bool razdeliVSkupine(int start, vector<vector<int>> &graf,
                     vector<int> &skupine) {
  queue<int> q;
  q.push(start);
  skupine[start] = 1;

  while (!q.empty()) {
    int otrok = q.front();
    q.pop();

    for (int sosed : graf[otrok]) {
      if (skupine[sosed] == 0) {
        skupine[sosed] = 3 - skupine[otrok];
        q.push(sosed);
      } else if (skupine[sosed] == skupine[otrok]) {
        return false;
      }
    }
  }

  return true;
}

int main() {
  int n, m;
  cin >> n >> m;

  vector<vector<int>> graf(n + 1);

  for (int i = 0; i < m; i++) {
    int a, b;
    cin >> a >> b;
    graf[a].push_back(b);
    graf[b].push_back(a);
  }

  vector<int> skupine(n + 1, 0);

  for (int i = 1; i <= n; i++) {
    if (skupine[i] == 0) {
      if (!razdeliVSkupine(i, graf, skupine)) {
        cout << -1 << endl;
        return 0;
      }
    }
  }

  for (int i = 1; i <= n; i++) {
    cout << skupine[i] << endl;
  }

  return 0;
}
