#include <iostream>
#include <algorithm>
#include <vector>
#include <stdbool.h>
#include <functional>
using namespace std;

int main() {
    int n, m;
    cin >> n >> m;

    vector<vector<bool>> adj_matrix(n, vector<bool>(n, false));
    vector<vector<int>> adj_list(n);

    for (int i = 0; i < m; ++i) {
        int u, v;
        cin >> u >> v;
        adj_matrix[u][v] = true;
        adj_matrix[v][u] = true;
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);
    }

    vector<vector<int>> independent_sets;

    for (int mask = 0; mask < (1 << n); ++mask) {
        vector<int> subset;
        for (int i = 0; i < n; ++i) {
            if (mask & (1 << i)) {
                subset.push_back(i);
            }
        }

        bool is_independent = true;
        for (int i = 0; i < subset.size(); ++i) {
            for (int j = i + 1; j < subset.size(); ++j) {
                int u = subset[i], v = subset[j];
                if (adj_matrix[u][v]) {
                    is_independent = false;
                    goto end_check;
                }
            }
        }

        end_check:
        if (is_independent) {
            independent_sets.push_back(subset);
        }
    }

    sort(independent_sets.begin(), independent_sets.end());

    for (auto& subset : independent_sets) {
        cout << "[";
        for (int i = 0; i < subset.size(); ++i) {
            cout << subset[i];
            if (i != subset.size() - 1) {
                cout << ", ";
            }
        }
        cout << "]" << endl;
    }

    bool is_complete = true;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (!adj_matrix[i][j]) {
                is_complete = false;
                goto end_complete_check;
            }
        }
    }
    end_complete_check:

    int chromatic;
    if (is_complete) {
        chromatic = n;
    } else {
        function<bool(int, int, vector<int>&)> canColor = [&](int k, int node, vector<int>& color) {
            if (node == n) return true;

            for (int c = 1; c <= k; ++c) {
                bool conflict = false;
                for (int neighbor : adj_list[node]) {
                    if (color[neighbor] == c) {
                        conflict = true;
                        break;
                    }
                }
                if (!conflict) {
                    color[node] = c;
                    if (canColor(k, node + 1, color)) return true;
                    color[node] = 0;
                }
            }
            return false;
        };

        chromatic = n;
        for (int k = 1; k <= n; ++k) {
            vector<int> color(n, 0);
            if (canColor(k, 0, color)) {
                chromatic = k;
                break;
            }
        }
    }

    cout << chromatic << endl;

    return 0;
}
