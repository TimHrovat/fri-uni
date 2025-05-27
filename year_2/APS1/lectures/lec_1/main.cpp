#include <iostream>
#include <vector>

using namespace std;

class BinaryHeap {
private:
  vector<int> t = {-1};

public:
  void push(int x) {
    t.push_back(x);

    int i = t.size() - 1;

    while (i > 1 && t[i] < t[i / 2]) {
      swap(t[i], t[i / 2]);
      i /= 2;
    }
  }

  int pop() {
    int x = t[1], i = 1;

    t[1] = t.back();

    t.pop_back();

    while (1) {
      int j = i;

      if (2 * i < t.size() && t[2 * i] < t[j])
        j = 2 * i;

      if (2 * i + 1 < t.size() && t[2 * i + 1] < t[j])
        j = 2 * i + 1;

      if (i == j)
        break;

      swap(t[i], t[j]);
      i = j;
    }

    return x;
  }
};

int main() {
  BinaryHeap h;
  vector<int> s = {6, 1, 2, 3, 0, 8, 5};

  for (int x : s) {
    h.push(x);
  }

  vector<int> u;

  for (int i = 0; i < s.size(); i++) {
    u.push_back(h.pop());
  }

  for (int num : u) {
    cout << num << "\n";
  }

  return 0;
}
