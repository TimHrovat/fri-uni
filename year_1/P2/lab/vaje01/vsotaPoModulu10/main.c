#include <stdio.h>

int getNum();

int main(int argc, char **argv) {
  int a = getNum();
  int b = getNum();

  printf("%d", (a + b) % 10);
  return 0;
}

int getNum() {
  int sign = 1;
  int number = 0;
  int chr = getchar();

  if (chr == '-') {
    sign = -1;
    chr = getchar();
  }

  while (chr != '\n' && chr != ' ') {
    number *= 10 + (chr - '0');

    chr = getchar();
  }

  return number;
}
