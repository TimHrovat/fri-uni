#include <iostream>
#include <vector>
#include <queue>
#include <set>

using namespace std;

const long long INF = LLONG_MAX;

long long dijkstra(int n, vector<vector<pair<int, int>>> &graf, int start, int end, pair<int, int> exclude = {-1, -1}) {
    vector<long long> dist(n, INF);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    dist[start] = 0;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [currDist, currNode] = pq.top();
        pq.pop();

        if (currDist > dist[currNode])
            continue;

        for (auto &[nextNode, weight] : graf[currNode]) {
            if (make_pair(currNode, nextNode) == exclude || make_pair(nextNode, currNode) == exclude)
                continue;

            long long newDist = currDist + weight;
            if (newDist < dist[nextNode]) {
                dist[nextNode] = newDist;
                pq.push({newDist, nextNode});
            }
        }
    }

    return dist[end];
}

long long drugaNajkrajsaPot(int n, vector<vector<pair<int, int>>> &graf) {
    vector<int> parent(n, -1);
    long long najkrajsaDolzina = dijkstra(n, graf, 0, n - 1);

    if (najkrajsaDolzina == INF)
        return -1;

    set<pair<int, int>> najkrajsaPovezava;

    vector<long long> dist(n, INF);
    priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<>> pq;
    dist[0] = 0;
    pq.push({0, 0});
    while (!pq.empty()) {
        auto [currDist, currNode] = pq.top();
        pq.pop();
        for (auto &[nextNode, weight] : graf[currNode]) {
            if (dist[currNode] + weight < dist[nextNode]) {
                dist[nextNode] = dist[currNode] + weight;
                parent[nextNode] = currNode;
                pq.push({dist[nextNode], nextNode});
            }
        }
    }

    int currNode = n - 1;
    while (parent[currNode] != -1) {
        int prevNode = parent[currNode];
        najkrajsaPovezava.insert({prevNode, currNode});
        najkrajsaPovezava.insert({currNode, prevNode}); // undirected graph
        currNode = prevNode;
    }

    long long drugaNajkrajsa = INF;
    for (auto &edge : najkrajsaPovezava) {
        long long novaDolzina = dijkstra(n, graf, 0, n - 1, edge);
        if (novaDolzina > najkrajsaDolzina && novaDolzina < drugaNajkrajsa) {
            drugaNajkrajsa = novaDolzina;
        }
    }

    return (drugaNajkrajsa == INF) ? -1 : drugaNajkrajsa;
}

int main() {
    int n, m;
    cin >> n >> m;

    vector<vector<pair<int, int>>> graf(n);

    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        graf[u].push_back({v, w});
        graf[v].push_back({u, w});
    }

    cout << drugaNajkrajsaPot(n, graf) << endl;
    return 0;
}
