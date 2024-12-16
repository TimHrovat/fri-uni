#include <iostream>
#include <sstream>
#include <string>

using namespace std;

typedef pair<int, int> PII;

std::pair<int, int> findSubtextWithWildcard(const std::string &text,
                                            const std::string &subtext) {
  int textLen = text.length();
  int subtextLen = subtext.length();

  for (int i = 0; i <= textLen - subtextLen; ++i) {
    bool match = true;

    for (int j = 0; j < subtextLen; ++j) {
      if (subtext[j] != '?' && text[i + j] != subtext[j]) {
        match = false;
        break;
      }
    }

    if (match) {
      return {i, i + subtextLen - 1};
    }
  }

  return {-1, -1};
}

bool isOnlyAsterisks(const std::string &str) {
  // Check if all characters in the string are '*'
  for (char c : str) {
    if (c != '*') {
      return false;
    }
  }
  return true;
}

PII findMatch(const string &pattern, const string &text) {
  stringstream ss(pattern);
  string subpattern;
  vector<string> subpatterns;

  if (isOnlyAsterisks(pattern)) {
    return {0, 0};
  }

  while (getline(ss, subpattern, '*')) {
    subpatterns.push_back(subpattern);
  }

  int textLen = text.length();
  int textIndex = 0;
  int start = -1, end = -1;

  for (size_t i = 0; i < subpatterns.size(); ++i) {
    const string &sub = subpatterns[i];
    if (sub.empty()) {
      if (i == 0)
        start = 0;
      continue;
    }

    auto found = findSubtextWithWildcard(text.substr(textIndex), sub);
    if (found.first == -1) {
      return {-1, -1};
    }

    int matchStart = textIndex + found.first;
    int matchEnd = textIndex + found.second;

    if (start == -1)
      start = matchStart;
    end = matchEnd;

    textIndex = matchEnd + 1;
  }

  if (!subpatterns.back().empty()) {
    return {start, end};
  } else {
    return {start, textLen - 1};
  }
}

int main() {
  int n;
  cin >> n;

  for (int i = 0; i < n; ++i) {
    string pattern, text;
    cin >> pattern >> text;

    auto match = findMatch(pattern, text);

    if (match.first < 0) {
      cout << -1 << endl;
    } else {
      cout << match.first << " " << match.second << endl;
    }
  }

  return 0;
}
