#include <cmath>
#include <iostream>
#include <random>

using namespace std;

class AVLNode {
public:
  int value;
  AVLNode *left, *right;
  int height;
  AVLNode(int v) : value(v), left(NULL), right(NULL), height(1) {}
};

class AVLTree {
public:
  AVLNode *root = NULL;

  int height() { return height(root); }
  int height(AVLNode *node) { return (node != NULL) ? node->height : 0; }

  int balance(AVLNode *node) {
    return height(node->right) - height(node->left);
  }

  void update(AVLNode *node) {
    node->height = 1 + fmax(height(node->left), height(node->right));
  }

  AVLNode *rotateLeft(AVLNode *node) {
    AVLNode *R = node->right;
    node->right = R->left;
    R->left = node;
    update(node);
    update(R);
    return R;
  }
  AVLNode *rotateRight(AVLNode *node) {
    AVLNode *L = node->left;
    node->left = L->right;
    L->right = node;
    update(node);
    update(L);
    return L;
  }

  bool contains(int x) { return contains(x, root); }
  bool contains(int x, AVLNode *node) {
    if (node == NULL)
      return false;
    else if (node->value == x)
      return true;
    else if (x < node->value)
      return contains(x, node->left);
    else
      return contains(x, node->right);
  }

  void insert(int x) { root = insert(x, root); }
  AVLNode *insert(int x, AVLNode *node) {
    if (node == NULL)
      return new AVLNode(x);

    if (x <= node->value)
      node->left = insert(x, node->left);
    else
      node->right = insert(x, node->right);

    update(node);

    int b = balance(node);

    // prevelik desni
    if (b==2) {
        int bR = balance(node->right);
        if (bR == 1 || bR == 0) return rotateLeft(node);
        else {
            node->right = rotateRight(node->right);
            return rotateLeft(node);
        }
    } else {
        int bL = balance(node->left);
        if (bL == -1 || bL == 0) return rotateRight(node);
        else {
            node->left = rotateLeft(node->left);
            return rotateRight(node);
        }
    }

    return node;
  }
};

int main() {
  default_random_engine rnd(123);
  AVLTree avl;
  int n = 1000;
  for (int i = 0; i < n; i++) {
    avl.insert(i);
  }
  cout << avl.height() << endl;
}
