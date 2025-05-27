#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

const int dx[] = {-1, 0, 1, 0};
const int dy[] = {0, 1, 0, -1};

class Solution {
private:
  int V, S;
  vector<vector<int>> heights;
  vector<vector<bool>> visited;

  bool isValid(int x, int y) { return x >= 0 && x < V && y >= 0 && y < S; }

  void dfs(int x, int y, int waterLevel) {
    visited[x][y] = true;

    for (int i = 0; i < 4; i++) {
      int newX = x + dx[i];
      int newY = y + dy[i];

      if (isValid(newX, newY) && !visited[newX][newY] &&
          heights[newX][newY] > waterLevel) {
        dfs(newX, newY, waterLevel);
      }
    }
  }

  int countIslands(int waterLevel) {
    int islands = 0;
    visited = vector<vector<bool>>(V, vector<bool>(S, false));

    for (int i = 0; i < V; i++) {
      for (int j = 0; j < S; j++) {
        if (!visited[i][j] && heights[i][j] > waterLevel) {
          dfs(i, j, waterLevel);
          islands++;
        }
      }
    }
    return islands;
  }

public:
  void solve() {
    cin >> V >> S;
    heights = vector<vector<int>>(V, vector<int>(S));

    int maxHeight = 0;
    for (int i = 0; i < V; i++) {
      for (int j = 0; j < S; j++) {
        cin >> heights[i][j];
        maxHeight = max(maxHeight, heights[i][j]);
      }
    }

    for (int waterLevel = 0; waterLevel <= maxHeight; waterLevel++) {
      cout << countIslands(waterLevel) << endl;
    }
  }
};

int main() {
  Solution solution;
  solution.solve();

  return 0;
}
