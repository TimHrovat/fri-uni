#include <iostream>
#include <vector>

using namespace std;

int main() {
  long long street_length, n;

  // left, right bounds
  vector<pair<long long, long long>> intervals;

  cin >> street_length >> n;

  for (int i = 0; i < n; i++) {
    long long location;
    long long strength;

    cin >> location >> strength;

    long long left = max(0LL, location - strength);
    long long right = min(street_length, location + strength);

    intervals.push_back({left, right});
  }

  sort(intervals.begin(), intervals.end());

  long long unlit_street_length = 0;
  long long last_end = 0;

  for (auto &interval : intervals) {
    if (last_end < interval.first) {
      unlit_street_length += interval.first - last_end;

      last_end = interval.second;
    } else {
      last_end = max(interval.second, last_end);
    }
  }

  unlit_street_length += street_length - last_end;

  cout << unlit_street_length << '\n';

  return 0;
}
