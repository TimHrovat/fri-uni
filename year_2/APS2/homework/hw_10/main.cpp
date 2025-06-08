#include <iostream>
#include <vector>
using namespace std;

const int INF = 10000000;

vector<vector<int>> initializeDp(int n, const vector<vector<int>> &cost);
void calculateDp(vector<vector<int>> &dp, int n,
                 const vector<vector<int>> &cost);
int findMinimumCycleCost(const vector<vector<int>> &dp, int n,
                         const vector<vector<int>> &cost);

int main() {
  int n;
  cin >> n;

  if (n == 1) {
    cout << 0 << endl;
    return 0;
  }

  vector<vector<int>> cost(n, vector<int>(n, 0));
  for (int i = 0; i < n - 1; i++) {
    for (int j = 0; j < n - 1 - i; j++) {
      int c;
      cin >> c;
      cost[i][i + 1 + j] = c;
      cost[i + 1 + j][i] = c;
    }
  }

  vector<vector<int>> dp = initializeDp(n, cost);
  calculateDp(dp, n, cost);
  int result = findMinimumCycleCost(dp, n, cost);

  cout << result << endl;
  return 0;
}

vector<vector<int>> initializeDp(int n, const vector<vector<int>> &cost) {
  int m = n - 1;
  int totalMasks = 1 << m;
  vector<vector<int>> dp(totalMasks, vector<int>(m, INF));

  for (int i = 0; i < m; i++) {
    dp[1 << i][i] = cost[0][i + 1];
  }

  return dp;
}

void calculateDp(vector<vector<int>> &dp, int n,
                 const vector<vector<int>> &cost) {
  int m = n - 1;
  int totalMasks = 1 << m;

  for (int mask = 0; mask < totalMasks; mask++) {
    for (int i = 0; i < m; i++) {
      if (!(mask & (1 << i)) || dp[mask][i] == INF)
        continue;

      for (int j = 0; j < m; j++) {
        if (mask & (1 << j))
          continue;
        int nextMask = mask | (1 << j);
        int nextCost = dp[mask][i] + cost[i + 1][j + 1];
        if (nextCost < dp[nextMask][j]) {
          dp[nextMask][j] = nextCost;
        }
      }
    }
  }
}

int findMinimumCycleCost(const vector<vector<int>> &dp, int n,
                         const vector<vector<int>> &cost) {
  int m = n - 1;
  int fullMask = (1 << m) - 1;
  int result = INF;

  for (int i = 0; i < m; i++) {
    if (dp[fullMask][i] == INF)
      continue;
    int cycleCost = dp[fullMask][i] + cost[i + 1][0];
    if (cycleCost < result) {
      result = cycleCost;
    }
  }

  return result;
}
