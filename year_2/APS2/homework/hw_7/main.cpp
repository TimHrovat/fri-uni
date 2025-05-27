#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

bool isPrime(int n);
int findP(int m);
vector<int> getPrimeFactors(int n);
int powMod(int a, int b, int mod);
vector<int> findPrimitiveRoots(int m, int p);

int main() {
  int m;
  cin >> m;
  int p = findP(m);
  vector<int> roots = findPrimitiveRoots(m, p);
  cout << p << ":";

  for (int root : roots) {
    cout << " " << root;
  }

  cout << endl;

  if (roots.empty()) {
    return 1;
  }

  int g = roots[0];

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      int exponent = i * j;
      int value = powMod(g, exponent, p);

      cout << value;

      if (j != m - 1) {
        cout << " ";
      }
    }

    cout << endl;
  }

  return 0;
}

bool isPrime(int n) {
  if (n <= 1) {
    return false;
  }

  if (n <= 3) {
    return true;
  }

  if (n % 2 == 0 || n % 3 == 0) {
    return false;
  }

  for (int i = 5; i * i <= n; i += 6) {
    if (n % i == 0 || n % (i + 2) == 0) {
      return false;
    }
  }

  return true;
}

int findP(int m) {
  int k = 1;

  while (true) {
    int candidate = k * m + 1;

    if (isPrime(candidate)) {
      return candidate;
    }

    k++;
  }
}

vector<int> getPrimeFactors(int n) {
  vector<int> factors;

  if (n % 2 == 0) {
    factors.push_back(2);

    while (n % 2 == 0) {
      n /= 2;
    }
  }
  for (int i = 3; i * i <= n; i += 2) {
    if (n % i == 0) {
      factors.push_back(i);

      while (n % i == 0) {
        n /= i;
      }
    }
  }

  if (n > 1) {
    factors.push_back(n);
  }

  return factors;
}

int powMod(int a, int b, int mod) {
  int result = 1;

  a %= mod;

  while (b > 0) {
    if (b % 2 == 1) {
      result = (result * a) % mod;
    }

    a = (a * a) % mod;
    b /= 2;
  }

  return result;
}

vector<int> findPrimitiveRoots(int m, int p) {
  if (m == 1) {
    return {1};
  }

  vector<int> factors_m = getPrimeFactors(m);
  vector<int> s_list;

  for (int s = 1; s < m; ++s) {
    bool coprime = true;

    for (int q : factors_m) {
      if (s % q == 0) {
        coprime = false;
        break;
      }
    }

    if (coprime) {
      s_list.push_back(s);
    }
  }

  vector<int> factors_p_minus_1 = getPrimeFactors(p - 1);
  int g = -1;

  for (int candidate = 2; candidate < p; ++candidate) {
    bool ok = true;

    for (int q : factors_p_minus_1) {
      int exponent = (p - 1) / q;

      if (powMod(candidate, exponent, p) == 1) {
        ok = false;
        break;
      }
    }

    if (ok) {
      g = candidate;
      break;
    }
  }

  if (g == -1) {
    return {};
  }

  int t = (p - 1) / m;
  int h = powMod(g, t, p);
  vector<int> roots;

  for (int s : s_list) {
    roots.push_back(powMod(h, s, p));
  }

  sort(roots.begin(), roots.end());

  return roots;
}
