#include <iostream>

using namespace std;

#define MAX_VAL 1000001

// s < 0 -> insert
// s = 0 -> delete
// s > 0 -> range (koliko elementov na temu range-u)

class Bag {
private:
  vector<int> seg_tree;
  vector<int> freq;

  void updateTree(int id, int val) { updateTree(id, val, 0, 0, MAX_VAL - 1); }

  void updateTree(int id, int val, int node, int start, int end) {
    if (start == end) {
      seg_tree[node] += val;
      return;
    }

    int mid = (start + end) / 2;
    int left = 2 * node + 1;
    int right = 2 * node + 2;

    if (id <= mid) {
      updateTree(id, val, left, start, mid);
    } else {
      updateTree(id, val, right, mid + 1, end);
    }

    seg_tree[node] = seg_tree[left] + seg_tree[right];
  }

  int query(int l, int r, int node, int start, int end) {
    if (r < start || l > end) {
      return 0;
    }

    if (l <= start && end <= r) {
      return seg_tree[node];
    }

    int mid = (start + end) / 2;
    int left = 2 * node + 1;
    int right = 2 * node + 2;

    int l_query = query(l, r, left, start, mid);
    int r_query = query(l, r, right, mid + 1, end);

    return l_query + r_query;
  }

public:
  Bag() {
    seg_tree.resize(4 * MAX_VAL, 0);
    freq.resize(MAX_VAL, 0);
  }

  void insert(int x) {
    freq[x]++;
    updateTree(x, 1);
  }

  void remove(int x) {
    if (freq[x] > 0) {
      freq[x]--;
      updateTree(x, -1);
    }
  }

  int query(int a, int b) { return query(a, b, 0, 0, MAX_VAL - 1); }
};

int main() {
  int n;
  cin >> n;

  Bag bag;
  long long res = 0;

  for (int i = 0; i < n; i++) {
    int s, x;
    cin >> s >> x;

    if (s < 0) {
      bag.insert(x);
    } else if (s == 0) {
      bag.remove(x);
    } else {
      res += bag.query(min(s, x), max(s, x));
    }
  }

  cout << res << endl;

  return 0;
}
