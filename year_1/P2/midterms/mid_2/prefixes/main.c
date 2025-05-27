#include <stdio.h>
#include <string.h>

int main() {
  char *str = "hello world";
  int len = strlen(str);
  char buf[strlen(str) + 1];

  for (int i = 0; i < len + 1; i++) {
    strncpy(buf, str, i);
    buf[i] = '\0';
    printf("%s\n", buf);
  }

  return 0;
}
