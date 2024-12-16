#include <cmath>
#include <iostream>
#include <sys/stdio.h>
#include <vector>

using namespace std;

void printCete(const vector<vector<int>> &);
vector<vector<int>> getCete(const vector<int> &);
vector<vector<int>> zlijCete(const vector<vector<int>> &, int);

int main() {
  // N - št elementov
  // K - št čet ki jih mergamo skupej
  // A - št korakov
  int n, k, a;

  cin >> n >> k >> a;

  vector<int> vec;

  for (int i = 0; i < n; i++) {
    int el;
    cin >> el;
    vec.push_back(el);
  }

  vector<vector<int>> cete = getCete(vec);

  for (int i = 0; i < a; i++) {
    if (cete.size() == 1)
      break;
    cete = zlijCete(cete, k);
  }

  printCete(cete);

  return 0;
}

vector<vector<int>> zlijCete(const vector<vector<int>> &cete, int k) {
  vector<vector<int>> zlite_cete;

  int skupine = ceil(cete.size() / (double)k);

  for (int i = 0; i < skupine; i++) {
    int num_of_cete =
        (i + 1 == skupine && cete.size() % k != 0) ? cete.size() % k : k;

    vector<int> indexi(num_of_cete, 0);

    int count = 0;
    for (int j = 0; j < num_of_cete; j++) {
      count += cete[i * k + j].size();
    }

    vector<int> zlitje;

    for (int j = 0; j < count; j++) {
      int min_val = INT_MAX;
      int min_index = -1;

      for (int z = 0; z < num_of_cete; z++) {
        if (indexi[z] < cete[i * k + z].size()) {
          if (cete[i * k + z][indexi[z]] < min_val) {
            min_val = cete[i * k + z][indexi[z]];
            min_index = z;
          }
        }
      }

      indexi[min_index]++;
      zlitje.push_back(min_val);
    }

    zlite_cete.push_back(zlitje);
  }

  return zlite_cete;
}

vector<vector<int>> getCete(const vector<int> &vec) {
  vector<vector<int>> cete;
  vector<int> cur_ceta;

  cur_ceta.push_back(vec[0]);

  for (int i = 1; i < vec.size(); i++) {
    if (vec[i] < vec[i - 1]) {
      cete.push_back(cur_ceta);
      cur_ceta.clear();
      cur_ceta.push_back(vec[i]);
    } else {
      cur_ceta.push_back(vec[i]);
    }
  }

  cete.push_back(cur_ceta);
  return cete;
}

void printCete(const vector<vector<int>> &cete) {
  for (const auto &ceta : cete) {
    for (int el : ceta) {
      cout << el << " ";
    }
  }
}
