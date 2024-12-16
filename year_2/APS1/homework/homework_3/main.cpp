#include <iostream>
#include <vector>
#include <stack>

using namespace std;

int main() {
  long long n, sum = 0;
  cin >> n;

  vector<long> heights(n);

  for (long long i = 0; i < n; i++) {
    cin >> heights[i];
  }

  // height, count
  stack<pair<long long, long long>> stack_left, stack_right;

  for (long long i = 0; i < n; i++) {
    int count = 0;

    while (!stack_left.empty() && stack_left.top().first <= heights[i]) {
      count += stack_left.top().second;
      stack_left.pop();
    }

    sum += count;

    stack_left.push({heights[i], count + 1});
  }

  for (long long i = n - 1; i >= 0; i--) {
    int count = 0;

    while (!stack_right.empty() && stack_right.top().first <= heights[i]) {
      count += stack_right.top().second;
      stack_right.pop();
    }

    sum += count;

    stack_right.push({heights[i], count + 1});
  }

  cout << sum << endl;

  return 0;
}
