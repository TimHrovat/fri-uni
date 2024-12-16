#include <iostream>
#include <queue>

using namespace std;

typedef long long int64;

#define N 1000
#define K 10

int n, e, k;

// vozl, cena
vector<pair<int, int>> adj[N + 1];

// št. vozlišč, št. nivojev
int64 dist[N + 1][K];

int64 inf = 1e15;

typedef array<int64, 3> Sol;

int main() {
  cin >> n >> e >> k;

  for (int x = 1; x <= n; x++) {
    for (int u = 0; u <= k; u++) {
      dist[x][u] = inf;
    }
  }

  for (int i = 0; i < e; i++) {
    int x, y, c;
    cin >> x >> y >> c;

    adj[x].push_back({y, c});
    adj[y].push_back({x, c});
  }

  priority_queue<Sol, vector<Sol>, greater<Sol>> pq;
  dist[1][0] = 0;

  pq.push({0, 1, 0});

  int64 res = -1;
  while (!pq.empty()) {
    auto [d, x, u] = pq.top();
    pq.pop();

    if (d != dist[x][u]) {
      continue;
    }

    if (x == n) {
      res = d;
      break;
    }

    for (auto [y, c] : adj[x]) {
      if (d + c < dist[y][u]) {
        dist[y][u] = d + c;
        pq.push({d + c, y, u});
      }

      // tale if dodamo dijkstri
      if (u < k && d + c / 2 < dist[y][u + 1]) {
        dist[y][u + 1] = d + c / 2;
        pq.push({d + c / 2, y, u + 1});
      }
    }
  }

  cout << res << endl;

  return 0;
}
