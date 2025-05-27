#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define MAX_TOKENS 100
#define MAX_LINE_LENGTH 1024

struct Token {
  const char *start;
  int length;
};

struct Token tokens[MAX_TOKENS];
int tokenCount = 0;

void tokenize(char *line);

int main() {
  int isInteractive = isatty(STDIN_FILENO);
  char line[MAX_LINE_LENGTH];

  while (1) {
    if (isInteractive) {
      printf("mysh> ");
      fflush(stdout);
    }

    if (fgets(line, sizeof(line), stdin) == NULL) {
      break;
    }

    char lineCopy[MAX_LINE_LENGTH];
    strcpy(lineCopy, line);

    size_t len = strlen(lineCopy);
    if (len > 0 && lineCopy[len - 1] == '\n') {
      lineCopy[len - 1] = '\0';
    }

    printf("Input line: '%s'\n", lineCopy);

    len = strlen(line);
    if (len > 0 && line[len - 1] == '\n') {
      line[len - 1] = '\0';
    }

    for (char *p = line; *p != '\0'; ++p) {
      if (*p == '#') {
        if (p == line || isspace(*(p - 1))) {
          *p = '\0';
          break;
        }
      }
    }

    int isEmpty = 1;
    for (char *p = line; *p != '\0'; ++p) {
      if (!isspace(*p)) {
        isEmpty = 0;
        break;
      }
    }

    if (isEmpty) {
      continue;
    }

    tokenize(line);

    for (int i = 0; i < tokenCount; ++i) {
      printf("Token %d: '%.*s'\n", i, tokens[i].length, tokens[i].start);
    }
  }

  return 0;
}

void tokenize(char *line) {
  tokenCount = 0;
  char *p = line;

  while (*p != '\0') {
    while (isspace(*p)) {
      p++;
    }

    if (*p == '\0') {
      break;
    }

    if (*p == '"') {
      p++;
      tokens[tokenCount].start = p;
      while (*p != '"' && *p != '\0') {
        p++;
      }
      tokens[tokenCount].length = p - tokens[tokenCount].start;
      tokenCount++;
      if (*p == '"') {
        p++;
      }
    } else {
      tokens[tokenCount].start = p;
      while (*p != '\0' && !isspace(*p)) {
        p++;
      }
      tokens[tokenCount].length = p - tokens[tokenCount].start;
      tokenCount++;
    }
  }
}
