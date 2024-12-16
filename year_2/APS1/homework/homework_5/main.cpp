#include <iostream>
#include <vector>
#include <string>

using namespace std;

class AutocompleteTrie {
private:
  char letter;
  int max_importance;
  int word_index;
  vector<AutocompleteTrie *> children;

public:
  AutocompleteTrie(char c = '\0') {
    letter = c;
    max_importance = -1;
    word_index = 0;
    children.resize(26, nullptr);
  }

  void insert(const string &word, int importance, int index, int pos = 0) {
    if (importance > max_importance) {
      max_importance = importance;
      word_index = index + 1;
    }

    if (pos == word.length()) {
      return;
    }

    int letter_index = word[pos] - 'a';

    if (children[letter_index] == nullptr) {
      children[letter_index] = new AutocompleteTrie(word[pos]);
    }

    children[letter_index]->insert(word, importance, index, pos + 1);
  }

  int autocomplete(const string &prefix, int pos = 0) {
    if (pos == prefix.length()) {
      return word_index;
    }

    int letter_index = prefix[pos] - 'a';

    if (children[letter_index] == nullptr) {
      return 0;
    }

    return children[letter_index]->autocomplete(prefix, pos + 1);
  }
};

AutocompleteTrie root;

int main() {
  int N;
  cin >> N;

  for (int i = 0; i < N; i++) {
    string word;
    int importance;

    cin >> word >> importance;

    root.insert(word, importance, i);
  }

  int N_PREFIXES;
  cin >> N_PREFIXES;
  vector<string> prefixes;

  for (int i = 0; i < N_PREFIXES; i++) {
    string prefix;

    cin >> prefix;

    prefixes.push_back(prefix);
  }

  for (auto prefix : prefixes) {
    cout << root.autocomplete(prefix) << '\n';
  }

  return 0;
}
