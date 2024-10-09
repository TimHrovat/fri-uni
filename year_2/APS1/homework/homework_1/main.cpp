#include <iostream>

using namespace std;

int main() {
  int dolzina, n;

  // lokacija, moč
  vector<pair<int, int>> svetilke;

  cin >> dolzina >> n;

  for (int i = 0; i < n; i++) {
    int lokacija;
    int moc;

    cin >> lokacija >> moc;

    svetilke.push_back({lokacija, moc});
  }

  return 0;
}
