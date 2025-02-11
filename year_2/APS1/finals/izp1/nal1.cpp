#include <iostream>
#include <queue>
#include <vector>

using namespace std;

double moc(double moc, vector<double> napoji) {
  priority_queue<double, vector<double>, greater<double>> pq(napoji.begin(),
                                                             napoji.end());

  while (moc >= pq.top()) {
    moc += pq.top();
    pq.pop();
  }

  return moc;
}

int main() {
  vector<double> napoji = {4, 20.7, 1, 5, 2.1};
  double x = moc(2.5, napoji);

  cout << x;
}
