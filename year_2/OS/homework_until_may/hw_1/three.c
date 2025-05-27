#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    {NULL, NULL, NULL}};

int main() {
  int isInteractive = isatty(STDIN_FILENO);
  char inputLine[MAX_LINE_LENGTH];
  char originalLine[MAX_LINE_LENGTH];

  while (1) {
    if (isInteractive) {
      printf("%s> ", shellPrompt);
      fflush(stdout);
    }

    if (fgets(inputLine, sizeof(inputLine), stdin) == NULL) {
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

    struct Command *command = getCommand(commandName);
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
