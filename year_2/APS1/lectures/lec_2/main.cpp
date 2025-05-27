#include <iostream>
#include <vector>
#include <cmath> // For pow and log2

using namespace std;

class RMQ {
private:
  int INF = 100000000;
  vector<int> array;
  int n;

  struct Node {
    int min, begin, end;
  };

  vector<Node> tree;

public:
  RMQ(vector<int> a) {
    n = pow(2, ceil(log2(a.size())));
    array = a;  // Initialize the class member array, not a new local variable
    array.resize(n, INF);
    tree.resize(2 * n);

    build();
  }

  void build(int id = 1) {
    if (id >= n) {
      int idx = id - n;
      tree[id] = {array[idx], idx, idx + 1};
      return;
    }

    int left = id * 2, right = id * 2 + 1;
    build(left);
    build(right);

    tree[id] = {
        min(tree[left].min, tree[right].min),
        tree[left].begin,
        tree[right].end,
    };
  }

  int query(int l, int r, int id = 1) {
    if (l <= tree[id].begin && tree[id].end <= r) { // Node interval within query interval
      return tree[id].min;
    }

    if (r <= tree[id].begin || tree[id].end <= l) { // Node interval outside query interval
      return INF;
    }

    return min(query(l, r, id * 2), query(l, r, id * 2 + 1));
  }
};

int main() {
  vector<int> v = {6, 3, 32, 1, 3, 4, 55, 67, 865, 21, 2, 4, 5, 1, 7, 3, 3};

  RMQ rmq(v);

  cout << rmq.query(1, 6);
  return 0;
}
