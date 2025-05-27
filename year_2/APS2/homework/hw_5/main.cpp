#include <climits>
#include <iostream>
#include <vector>

using namespace std;

void printSubarray(const vector<int> &arr, int left, int right) {
  cout << "[";
  for (int i = left; i <= right; ++i) {
    cout << arr[i];
    if (i < right)
      cout << ", ";
  }
  cout << "]: ";
}

int maxCrossingSubarray(const vector<int> &arr, int left, int mid, int right) {
  int leftSum = INT_MIN;
  int sum = 0;

  for (int i = mid; i >= left; --i) {
    sum += arr[i];
    if (sum > leftSum) {
      leftSum = sum;
    }
  }

  int rightSum = INT_MIN;
  sum = 0;

  for (int i = mid + 1; i <= right; ++i) {
    sum += arr[i];
    if (sum > rightSum) {
      rightSum = sum;
    }
  }

  return leftSum + rightSum;
}

int maxSubarray(const vector<int> &arr, int left, int right) {
  if (left == right) {
    printSubarray(arr, left, right);
    cout << arr[left] << endl;
    return arr[left];
  }

  int mid = (left + right) / 2;

  int leftMax = maxSubarray(arr, left, mid);
  int rightMax = maxSubarray(arr, mid + 1, right);
  int crossMax = maxCrossingSubarray(arr, left, mid, right);

  printSubarray(arr, left, right);
  cout << max(max(leftMax, rightMax), crossMax) << endl;

  return max(max(leftMax, rightMax), crossMax);
}

int main() {
  vector<int> arr;
  int x;
  while (cin >> x) {
    arr.push_back(x);
  }

  if (!arr.empty()) {
    maxSubarray(arr, 0, arr.size() - 1);
  }

  return 0;
}
