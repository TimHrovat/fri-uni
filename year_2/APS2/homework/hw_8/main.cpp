#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

string roundNumber(double number);
string complexToString(double real, double imag);
vector<complex<double>> fft(vector<complex<double>> a, bool invert,
                            vector<string> &trace);

int main() {
  int n;
  cin >> n;

  vector<double> a_coeffs(n), b_coeffs(n);
  for (int i = 0; i < n; ++i) {
    cin >> a_coeffs[i];
  }
  for (int i = 0; i < n; ++i) {
    cin >> b_coeffs[i];
  }

  int product_size = 2 * n - 1;
  int m = 1;
  while (m < product_size) {
    m <<= 1;
  }

  vector<complex<double>> a(m, 0.0), b(m, 0.0);
  for (int i = 0; i < n; ++i) {
    a[i] = complex<double>(a_coeffs[i], 0.0);
    b[i] = complex<double>(b_coeffs[i], 0.0);
  }

  vector<string> trace_fft_a;
  auto fft_a = fft(a, false, trace_fft_a);

  vector<string> trace_fft_b;
  auto fft_b = fft(b, false, trace_fft_b);

  vector<complex<double>> c(m);
  for (int i = 0; i < m; ++i) {
    c[i] = fft_a[i] * fft_b[i];
  }

  vector<string> trace_inv_fft;
  auto inv_fft = fft(c, true, trace_inv_fft);

  vector<complex<double>> product(m);
  for (int i = 0; i < m; ++i) {
    product[i] = inv_fft[i] / (double)m;
  }

  for (const string &line : trace_fft_a) {
    cout << line << endl;
  }

  for (const string &line : trace_fft_b) {
    cout << line << endl;
  }

  for (const string &line : trace_inv_fft) {
    cout << line << endl;
  }

  for (int i = 0; i < m; ++i) {
    cout << complexToString(product[i].real(), product[i].imag());
    if (i < m - 1) {
      cout << " ";
    }
  }
  cout << endl;

  return 0;
}

string roundNumber(double number) {
  stringstream output;
  output << fixed << setprecision(5) << number;
  string str = output.str();
  size_t ix = str.find(',');
  if (ix != string::npos) {
    str.replace(ix, 1, ".");
  }
  int i = str.size() - 1;
  while (i >= 0 && str[i] == '0') {
    i--;
  }
  if (i >= 0 && str[i] == '.') {
    return str.substr(0, i + 2);
  }
  if (i < 0) {
    return "0.0";
  }
  return str.substr(0, i + 1);
}

string complexToString(double real, double imag) {
  const double EPS = 1e-12;
  bool has_real = abs(real) >= EPS;
  bool has_imag = abs(imag) >= EPS;
  if (!has_real && !has_imag) {
    return "0.0";
  }
  if (has_real && !has_imag) {
    return roundNumber(real);
  }
  if (!has_real && has_imag) {
    return roundNumber(imag) + "i";
  }
  char sign = (imag > 0) ? '+' : '-';
  return roundNumber(real) + sign + roundNumber(fabs(imag)) + "i";
}

vector<complex<double>> fft(vector<complex<double>> a, bool invert,
                            vector<string> &trace) {
  int n = a.size();
  if (n == 1) {
    return a;
  }

  vector<complex<double>> even(n / 2), odd(n / 2);
  for (int i = 0; i < n / 2; ++i) {
    even[i] = a[2 * i];
    odd[i] = a[2 * i + 1];
  }

  even = fft(even, invert, trace);
  odd = fft(odd, invert, trace);

  double angle = 2 * M_PI / n * (invert ? -1 : 1);
  complex<double> w(cos(angle), sin(angle));
  complex<double> wk(1.0, 0.0);

  vector<complex<double>> y(n);
  for (int k = 0; k < n / 2; ++k) {
    complex<double> t = wk * odd[k];
    y[k] = even[k] + t;
    y[k + n / 2] = even[k] - t;
    wk *= w;
  }

  if (n >= 2) {
    string line;
    for (int i = 0; i < n; ++i) {
      line += complexToString(y[i].real(), y[i].imag());
      if (i != n - 1) {
        line += " ";
      }
    }
    trace.push_back(line);
  }

  return y;
}
