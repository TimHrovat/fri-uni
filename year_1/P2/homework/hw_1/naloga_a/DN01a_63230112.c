#include <stdio.h>
#include <stdlib.h>

long long getLongLong();
void printLongLong(long long  num);
void printLongLongRec(long long num);

int main(int argc, char **argv) {
  long long a = getLongLong();
  long long b = getLongLong();

  printLongLong(a + b);
  return 0;
}

long long getLongLong() {
  int sign = 1;
  int number = 0;
  int chr = getchar();

  if (chr == '-') {
    sign = -1;
    chr = getchar();
  }

  while (chr != '\n' && chr != ' ') {
    number = number * 10 + (chr - '0');

    chr = getchar();
  }

  return number * sign;
}

void printLongLong(long long num) {
    if (num == 0) {
        putchar('0');
        putchar('\n');
        return;
    }

    if (num < 0) {
        putchar('-');
    }

    printLongLongRec(num);
    putchar('\n');
}

void printLongLongRec(long long num) {
    if (num == 0) {
        return; 
    }

    printLongLongRec(num / 10);
    
    putchar('0' + (llabs(num % 10)));
}
