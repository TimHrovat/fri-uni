#include <iostream>
#include <vector>

using namespace std;

struct Oseba {
  int id;
  int povisica;
  vector<Oseba *> children;
};

Oseba *buildTree(int N, const vector<int> &managers,
                 const vector<int> &povisice) {
  vector<Oseba *> vec(N, nullptr);

  for (int i = 0; i < N; i++) {
    vec[i] = new Oseba({i, povisice[i]});
  }

  for (int i = 0; i < N; i++) {
    if (managers[i] != 0) {
      vec[managers[i]]->children.push_back(vec[i]);
    }
  }

  return vec[0];
}

Oseba *search(Oseba &root, int id);

int main() {
  int N;

  cin >> N;

  vector<int> managers(N);
  vector<int> povisice(N);

  for (int i = 0; i < N; i++) {
    cin >> managers[i];
  }

  for (int i = 0; i < N; i++) {
    cin >> povisice[i];
  }

  buildTree(N, managers, povisice);

  return 0;
}
