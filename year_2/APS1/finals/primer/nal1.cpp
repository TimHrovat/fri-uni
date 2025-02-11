/**
 * Najprej iteriramo čez daljice in jih zapišemo v vektor vektorjev:
 *   - index = prva točka, vrednost = druga točka
 *   - index = druga točka, vrednost = prva točka 
 * Nastavi začetno točko (point) na prvo točko prve daljice in prejšnjo točko (prev_point na -1)
 * Pregleda dve sosednji točki trenutne in izbere tisto, ki ni enaka prev_point
 * Shrani novo daljico ({prev_point, point}) v rezultat in premakne prev_point ter point naprej
 * Ustavi, ko se trenutna točka (point) ujema z začetno točko prve daljice
 *
 * prostorska zahtevnost: O(d) 
 * časovna zahtevnost: O(n + d)
 */
#include <iostream>
#include <vector>

using namespace std;

typedef pair<int, int> PII;

vector<PII> ovojnica(int st_tock, vector<PII> daljice);

int main() {
  vector<PII> daljice = {{2, 5}, {7, 10}, {3, 5}, {3, 10}, {7, 2}};
  vector<PII> krog = ovojnica(12, daljice);

  cout << "(";
  for (auto k : krog) {
    cout << "(" << k.first << ", " << k.second << "),";
  }
  cout << ")";
}

vector<PII> ovojnica(int n, vector<PII> daljice) {
    vector<vector<int>> order(n + 1);
    for (PII daljica: daljice) {
        order[daljica.first].push_back(daljica.second);
        order[daljica.second].push_back(daljica.first);
    }
    vector<PII> result;
    int prev_point = -1;
    int point = daljice[0].first;
    while (true) {
        if (order[point][0] != prev_point) {
            prev_point = point;
            point = order[point][0];
        } else {
            prev_point = point;
            point = order[point][1];    
        }
        result.push_back({prev_point, point});
        if (point == result[0].first) {
            break;
        }
    }
    return result;
}
