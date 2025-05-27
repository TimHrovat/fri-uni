#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <libgen.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_TOKENS 100
#define MAX_LINE_LENGTH 1024
#define MAX_PROMPT_LENGTH 8

struct Token {
  const char *start;
  int length;
};

struct Command {
  const char *name;
  int (*handler)(int argc, char **argv);
  const char *description;
};

void tokenizeInput(char *inputLine);
struct Command *getCommand(const char *commandName);
int builtin_debug(int argc, char **argv);
int builtin_prompt(int argc, char **argv);
int builtin_status(int argc, char **argv);
int builtin_exit(int argc, char **argv);
int builtin_help(int argc, char **argv);
int builtin_print(int argc, char **argv);
int builtin_echo(int argc, char **argv);
int builtin_len(int argc, char **argv);
int builtin_sum(int argc, char **argv);
int builtin_calc(int argc, char **argv);
int builtin_basename(int argc, char **argv);
int builtin_dirname(int argc, char **argv);

struct Token tokens[MAX_TOKENS];
int commandTokenCount = 0;
int debugLevel = 0;
int lastExitStatus = 0;
char shellPrompt[MAX_PROMPT_LENGTH + 1] = "mysh";

struct Command commands[] = {
    {"debug", builtin_debug, "Set or show debug level"},
    {"prompt", builtin_prompt, "Set or show the prompt (max 8 chars)"},
    {"status", builtin_status, "Show the exit status of the last command"},
    {"exit", builtin_exit, "Exit the shell"},
    {"help", builtin_help, "Show this help message"},
    {"print", builtin_print},
    {"echo", builtin_echo},
    {"len", builtin_len},
    {"sum", builtin_sum},
    {"calc", builtin_calc},
    {"basename", builtin_basename},
    {"dirname", builtin_dirname},
    {NULL, NULL, NULL}};

int builtin_debug(int argc, char **argv) {
  if (argc == 1) {
    printf("%d\n", debugLevel);
    return 0;
  } else {
    char *endptr;
    long level = strtol(argv[1], &endptr, 10);
    if (*endptr != '\0' || argv[1][0] == '\0') {
      debugLevel = 0;
    } else {
      debugLevel = (int)level;
    }
    return 0;
  }
}

int builtin_prompt(int argc, char **argv) {
  if (argc == 1) {
    printf("%s\n", shellPrompt);
    return 0;
  } else {
    if (strlen(argv[1]) > MAX_PROMPT_LENGTH) {
      return 1;
    }
    strncpy(shellPrompt, argv[1], MAX_PROMPT_LENGTH);
    shellPrompt[MAX_PROMPT_LENGTH] = '\0';
    return 0;
  }
}

int builtin_status(int argc, char **argv) {
  printf("%d\n", lastExitStatus);
  return 0;
}

int builtin_exit(int argc, char **argv) {
  int status = lastExitStatus;
  if (argc >= 2) {
    char *endptr;
    long val = strtol(argv[1], &endptr, 10);
    if (*endptr == '\0' && argv[1][0] != '\0') {
      status = (int)val;
    }
  }
  exit(status);
}

int builtin_help(int argc, char **argv) {
  printf("Built-in commands:\n");
  for (struct Command *cmd = commands; cmd->name != NULL; cmd++) {
    printf("  %s: %s\n", cmd->name, cmd->description);
  }
  return 0;
}

int builtin_print(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (i > 1)
      printf(" ");
    printf("%s", argv[i]);
  }
  fflush(stdout);
  return 0;
}

int builtin_echo(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (i > 1)
      printf(" ");
    printf("%s", argv[i]);
  }
  printf("\n");
  fflush(stdout);
  return 0;
}

int builtin_len(int argc, char **argv) {
  int total = 0;
  for (int i = 1; i < argc; i++) {
    total += strlen(argv[i]);
  }
  printf("%d\n", total);
  fflush(stdout);
  return 0;
}

int builtin_sum(int argc, char **argv) {
  int sum = 0;
  for (int i = 1; i < argc; i++) {
    sum += atoi(argv[i]);
  }
  printf("%d\n", sum);
  fflush(stdout);
  return 0;
}

int builtin_calc(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "calc: expects three arguments\n");
    fflush(stderr);
    return 1;
  }

  char *op = argv[2];
  if (strlen(op) != 1 || strchr("+-*/%", op[0]) == NULL) {
    fprintf(stderr, "calc: invalid operator\n");
    fflush(stderr);
    return 1;
  }

  char *end;
  long a = strtol(argv[1], &end, 10);
  if (*end != '\0' || errno == ERANGE) {
    fprintf(stderr, "calc: invalid number '%s'\n", argv[1]);
    fflush(stderr);
    return 1;
  }

  long b = strtol(argv[3], &end, 10);
  if (*end != '\0' || errno == ERANGE) {
    fprintf(stderr, "calc: invalid number '%s'\n", argv[3]);
    fflush(stderr);
    return 1;
  }

  switch (op[0]) {
  case '+':
    printf("%ld\n", a + b);
    break;
  case '-':
    printf("%ld\n", a - b);
    break;
  case '*':
    printf("%ld\n", a * b);
    break;
  case '/':
    if (b == 0) {
      fprintf(stderr, "calc: division by zero\n");
      fflush(stderr);
      return 1;
    }
    printf("%ld\n", a / b);
    break;
  case '%':
    if (b == 0) {
      fprintf(stderr, "calc: modulo by zero\n");
      fflush(stderr);
      return 1;
    }
    printf("%ld\n", a % b);
    break;
  }

  fflush(stdout);
  return 0;
}

int builtin_basename(int argc, char **argv) {
  if (argc != 2) {
    fflush(stderr);
    return 1;
  }

  char *path = strdup(argv[1]);
  if (!path) {
    perror("basename");
    return 1;
  }

  char *base = basename(path);
  printf("%s\n", base);
  free(path);
  fflush(stdout);
  return 0;
}

int builtin_dirname(int argc, char **argv) {
  if (argc != 2) {
    fflush(stderr);
    return 1;
  }

  char *path = strdup(argv[1]);
  if (!path) {
    perror("dirname");
    return 1;
  }

  char *dir = dirname(path);
  printf("%s\n", dir);
  free(path);
  fflush(stdout);
  return 0;
}

int main() {
  int isInteractive = isatty(STDIN_FILENO);
  char inputLine[MAX_LINE_LENGTH];
  char originalLine[MAX_LINE_LENGTH];
  struct Command *command = NULL;

  while (1) {
    if (isInteractive) {
      printf("%s> ", shellPrompt);
      fflush(stdout);
    }

    if (fgets(inputLine, sizeof(inputLine), stdin) == NULL) {
      if (strcmp(command->name, "status") == 0) {
        printf("Exit status: %d", lastExitStatus);
        fflush(stdout);
      }
      break;
    }

    strcpy(originalLine, inputLine);
    size_t lineLength = strlen(originalLine);
    if (lineLength > 0 && originalLine[lineLength - 1] == '\n') {
      originalLine[lineLength - 1] = '\0';
    }

    lineLength = strlen(inputLine);
    if (lineLength > 0 && inputLine[lineLength - 1] == '\n') {
      inputLine[lineLength - 1] = '\0';
    }

    for (char *p = inputLine; *p != '\0'; ++p) {
      if (*p == '#' && (p == inputLine || isspace(*(p - 1)))) {
        *p = '\0';
        break;
      }
    }

    bool isLineEmpty = true;
    for (char *p = inputLine; *p != '\0'; ++p) {
      if (!isspace(*p)) {
        isLineEmpty = false;
        break;
      }
    }

    if (debugLevel > 0) {
      printf("Input line: '%s'\n", originalLine);
    }

    if (isLineEmpty) {
      continue;
    }

    tokenizeInput(inputLine);

    if (debugLevel > 0) {
      for (int i = 0; i < commandTokenCount; ++i) {
        printf("Token %d: '%.*s'\n", i, tokens[i].length, tokens[i].start);
      }
    }

    const char *inputRedirect = NULL;
    const char *outputRedirect = NULL;
    bool runInBackground = false;
    int commandTokenCount = commandTokenCount;

    if (commandTokenCount > 0) {
      struct Token lastToken = tokens[commandTokenCount - 1];
      if (lastToken.length == 1 && *lastToken.start == '&') {
        runInBackground = true;
        commandTokenCount--;
      }
    }

    if (commandTokenCount > 0) {
      struct Token lastToken = tokens[commandTokenCount - 1];
      if (lastToken.start[0] == '>' && lastToken.length >= 1) {
        outputRedirect = lastToken.start + 1;
        commandTokenCount--;
      }
    }

    if (commandTokenCount > 0) {
      struct Token lastToken = tokens[commandTokenCount - 1];
      if (lastToken.start[0] == '<' && lastToken.length >= 1) {
        inputRedirect = lastToken.start + 1;
        commandTokenCount--;
      }
    }

    if (debugLevel > 0) {
      if (inputRedirect) {
        printf("Input redirect: '%.*s'\n", (int)(strlen(inputRedirect)),
               inputRedirect);
      }
      if (outputRedirect) {
        printf("Output redirect: '%.*s'\n", (int)(strlen(outputRedirect)),
               outputRedirect);
      }
      if (runInBackground) {
        printf("Background: 1\n");
      }
    }

    if (commandTokenCount == 0) {
      continue;
    }

    char commandName[MAX_LINE_LENGTH];
    snprintf(commandName, tokens[0].length + 1, "%.*s", tokens[0].length,
             tokens[0].start);

    command = getCommand(commandName);
    if (command != NULL) {
      char *argv[MAX_TOKENS + 1];
      for (int i = 0; i < commandTokenCount; ++i) {
        argv[i] = strndup(tokens[i].start, tokens[i].length);
      }
      argv[commandTokenCount] = NULL;

      if (debugLevel > 0) {
        printf("Executing builtin '%s' in %s\n", commandName,
               runInBackground ? "background" : "foreground");
      }

      int previousStatus = lastExitStatus;
      int result = command->handler(commandTokenCount, argv);

      if (strcmp(commandName, "status") != 0) {
        lastExitStatus = result;
      } else {
        lastExitStatus = previousStatus;
      }

      for (int i = 0; i < commandTokenCount; ++i) {
        free(argv[i]);
      }
    } else {
      if (debugLevel > 0) {
        char externalCommand[MAX_LINE_LENGTH] = "";
        for (int i = 0; i < commandTokenCount; ++i) {
          if (i > 0)
            strcat(externalCommand, " ");
          strncat(externalCommand, tokens[i].start, tokens[i].length);
        }
        printf("External command '%s'\n", externalCommand);
      }
      lastExitStatus = 0;
    }
  }

  return 0;
}

void tokenizeInput(char *inputLine) {
  commandTokenCount = 0;
  char *p = inputLine;
  while (*p != '\0' && commandTokenCount < MAX_TOKENS) {
    while (isspace(*p)) {
      p++;
    }
    if (*p == '\0')
      break;

    if (*p == '"') {
      p++;
      tokens[commandTokenCount].start = p;
      while (*p != '"' && *p != '\0') {
        p++;
      }
      tokens[commandTokenCount].length = p - tokens[commandTokenCount].start;
      commandTokenCount++;
      if (*p == '"')
        p++;
    } else {
      tokens[commandTokenCount].start = p;
      while (*p != '\0' && !isspace(*p)) {
        p++;
      }
      tokens[commandTokenCount].length = p - tokens[commandTokenCount].start;
      commandTokenCount++;
    }
  }
}

struct Command *getCommand(const char *commandName) {
  for (struct Command *cmd = commands; cmd->name != NULL; cmd++) {
    if (strncmp(cmd->name, commandName, strlen(cmd->name)) == 0 &&
        strlen(cmd->name) == strlen(commandName)) {
      return cmd;
    }
  }
  return NULL;
}
