#include <iostream>
#include <vector>

using namespace std;

int main() {
  long long n;
  cin >> n;

  vector<long long> heights, sums(n);
  long long final_sum = 0;

  for (long long i = 0; i < n; i++) {
    long long tmp;
    cin >> tmp;
    heights.push_back(tmp);
  }

  long long max_h_left = -1;
  for (long long i = 0; i < heights.size(); i++) {
    long long height = heights[i];

    long long sum = 0;

    for (long long j = i - 1; j > max_h_left; j--) {
      if (heights[j] > height) {
        break;
      }

      sum++;
    }

    if (max_h_left != -1 && heights[max_h_left] <= height) {
      sum = sums[max_h_left] + 1;
    }

    if (max_h_left == -1 || heights[max_h_left] <= height) {
      max_h_left = i;
    }

    for (long long j = i + 1; j < n; j++) {
      if (heights[j] > height) {
        break;
      }

      sum++;
    }

    sums[i] = sum;
    final_sum += sum;
  }

  cout << final_sum;

  return 0;
}
