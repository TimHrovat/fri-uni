#include <iostream>
#include <queue>
#include <vector>

using namespace std;

priority_queue<int> manjsa_st;
priority_queue<int, vector<int>, greater<int>> vecja_st;

void insert(int n) {
  if (manjsa_st.empty() || n < manjsa_st.top()) {
    manjsa_st.push(n);
  } else {
    vecja_st.push(n);
  }

  if (manjsa_st.size() > vecja_st.size() + 1) {
    vecja_st.push(manjsa_st.top());
    manjsa_st.pop();
  } else if (vecja_st.size() > manjsa_st.size()) {
    manjsa_st.push(vecja_st.top());
    vecja_st.pop();
  }
}

int main() {
  int N;
  cin >> N;

  for (int i = 0; i < N; i++) {
    int x;
    cin >> x;

    insert(x);
    cout << manjsa_st.top() << "\n";
  }

  return 0;
}
