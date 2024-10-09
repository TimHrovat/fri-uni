#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

// how to test
// export name=main

// how to compile
// g++ -std=c++17 -lm resitev.cpp -o resitev

int main() {
  int n, k;

  cin >> n >> k;

  vector<pair<int, string>> osebe;

  for (int i = 0; i < n; i++) {
    string ime;
    int starost;

    cin >> ime >> starost;

    osebe.push_back({starost, ime});
  }

  // sort ureja po prvem elementu v paru v tem primeru int
  sort(osebe.begin(), osebe.end());
  reverse(osebe.begin(), osebe.end());

  vector<string> imena;

  for (int i = 0; i < k; i++) {
    auto [starost, ime] = osebe[i];

    imena.push_back(ime);
  }

  sort(imena.begin(), imena.end());

  for (auto ime : imena) {
    cout << ime << "\n";
  }

  return 0;
}
