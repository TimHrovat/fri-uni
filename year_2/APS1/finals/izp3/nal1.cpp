/**
 * 1) uporabimo min-heap za efektivno iskanje najmanjših števil
 *
 * Uporabimo priority queue iz katerega lahko vedno vzamemo najmanjše število in ga odstranimo
 * nato vzamemo in odstranimo še drugo najmanjše število
 * tako imamo dve najmanjši števili ki ju seštejemo in dodamo nazaj v queue
 * na koncu vrnemo zadnji dve številki za kateri smo izračunali vsoto
 *
 * č: O(nlogn), potrebujemo O(nlogn) za inicializacijo kupa in O(logn) za operacije na kupu
 * p: O(n), ker shranimo vse elemente na kupu
 */
#include <iostream>
#include <queue>
#include <vector>

using namespace std;

pair<int, int> vsota(vector<int> &stevila);

int main() {
  vector<int> stevila = {3, 5, 3, 1, 6};
  pair<int, int> zadnji = vsota(stevila);

  cout << zadnji.first << ", " << zadnji.second;
}

pair<int, int> vsota(vector<int> &stevila) {
  priority_queue<int, vector<int>, greater<int>> heap(stevila.begin(),
                                                      stevila.end());

  pair<int, int> zadnji;

  while (heap.size() > 1) {
    int prvi_min = heap.top();
    heap.pop();
    int drugi_min = heap.top();
    heap.pop();

    zadnji = {prvi_min, drugi_min};

    heap.push(prvi_min + drugi_min);
  }

  return zadnji;
}
