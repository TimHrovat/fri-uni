#include <iostream>
#include <map>
#include <queue>
#include <set>

using namespace std;

typedef vector<pair<int, int>> VII;
typedef struct {
  int id;
  VII cons;
} Node;
typedef struct {
  int cost, node, n_kupons;
} PriorityItem;

// N -> končno vozlišče
// E -> št. vrstic
// K -> št. kuponov
int N, E, K;

PriorityItem findMin(map<int, Node> graph, set<int> razvita,
                     priority_queue<PriorityItem> pq) {
  auto top = pq.top();
  pq.pop();

  auto graph_node = graph[top.node];

  if (razvita.find(N) != razvita.end()) {
    return top;
  }

  if (razvita.find(graph_node.id) == razvita.end()) {
    return findMin(graph, razvita, pq);
  } else {
    for (auto &con : graph_node.cons) {
      if (razvita.find(con.first) == razvita.end()) {
        continue;
      }

      if (top.n_kupons < K) {
          pq.push({con.first, top.cost + con.second, top.n_kupons + 1});
      }

      pq.push({con.first, top.cost + con.second, top.n_kupons});
    }

    return findMin(graph, razvita, pq);
  }
}

int main() {
  map<int, Node> graph;

  cin >> N >> E >> K;

  for (int i = 0; i < E; i++) {
    int x, y, cost;
    cin >> x >> y >> cost;

    auto node = graph[x];

    node.id = x;
    node.cons.push_back({y, cost});
  }

  set<int> razvita;
  priority_queue<PriorityItem> pq;
  pq.push({0, 1, 0});

  auto f = findMin(graph, razvita, pq);

  cout << f.cost << endl;

  return 0;
}
