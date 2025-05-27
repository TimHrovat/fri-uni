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
#include <sys/utsname.h>
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
int builtin_dirch(int argc, char **argv);
int builtin_dirwd(int argc, char **argv);
int builtin_dirmk(int argc, char **argv);
int builtin_dirrm(int argc, char **argv);
int builtin_dirls(int argc, char **argv);
int builtin_rename(int argc, char **argv);
int builtin_unlink(int argc, char **argv);
int builtin_remove(int argc, char **argv);
int builtin_linkhard(int argc, char **argv);
int builtin_linksoft(int argc, char **argv);
int builtin_linkread(int argc, char **argv);
int builtin_linklist(int argc, char **argv);
int builtin_cpcat(int argc, char **argv);
int builtin_pid(int argc, char **argv);
int builtin_ppid(int argc, char **argv);
int builtin_uid(int argc, char **argv);
int builtin_euid(int argc, char **argv);
int builtin_gid(int argc, char **argv);
int builtin_egid(int argc, char **argv);
int builtin_sysinfo(int argc, char **argv);

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
    {"dirch", builtin_dirch},
    {"dirwd", builtin_dirwd},
    {"dirmk", builtin_dirmk},
    {"dirrm", builtin_dirrm},
    {"dirls", builtin_dirls},
    {"rename", builtin_rename},
    {"unlink", builtin_unlink},
    {"remove", builtin_remove},
    {"linkhard", builtin_linkhard},
    {"linksoft", builtin_linksoft},
    {"linkread", builtin_linkread},
    {"linklist", builtin_linklist},
    {"cpcat", builtin_cpcat},
    {"pid", builtin_pid},
    {"ppid", builtin_ppid},
    {"uid", builtin_uid},
    {"euid", builtin_euid},
    {"gid", builtin_gid},
    {"egid", builtin_egid},
    {"sysinfo", builtin_sysinfo},
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
  fflush(stdout);
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

int builtin_dirch(int argc, char **argv) {
  const char *dir = argc >= 2 ? argv[1] : "/";
  if (chdir(dir) == -1) {
    int saved_errno = errno;
    perror("dirch");
    return saved_errno;
  }
  return 0;
}

int builtin_dirwd(int argc, char **argv) {
  char path[PATH_MAX];
  if (getcwd(path, sizeof(path)) == NULL) {
    int saved_errno = errno;
    perror("dirwd");
    return saved_errno;
  }

  const char *mode = "base";
  if (argc >= 2)
    mode = argv[1];

  if (strcmp(mode, "full") == 0) {
    printf("%s\n", path);
  } else if (strcmp(mode, "base") == 0) {
    printf("%s\n", basename(path));
  } else {
    fprintf(stderr, "dirwd: invalid mode '%s'\n", mode);
    fflush(stderr);
    return 1;
  }
  fflush(stdout);
  return 0;
}

int builtin_dirmk(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "dirmk: missing operand\n");
    fflush(stderr);
    return 1;
  }
  if (mkdir(argv[1], 0755) == -1) {
    int saved_errno = errno;
    perror("dirmk");
    return saved_errno;
  }
  return 0;
}

int builtin_dirrm(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "dirrm: missing operand\n");
    fflush(stderr);
    return 1;
  }
  if (rmdir(argv[1]) == -1) {
    int saved_errno = errno;
    perror("dirrm");
    return saved_errno;
  }
  return 0;
}

int builtin_dirls(int argc, char **argv) {
  const char *path = argc >= 2 ? argv[1] : ".";
  struct dirent **entries;
  int n = scandir(path, &entries, NULL, NULL);
  if (n == -1) {
    int saved_errno = errno;
    perror("dirls");
    return saved_errno;
  }

  int first = 1;
  for (int i = 0; i < n; i++) {
    if (!first)
      printf("  ");
    printf("%s", entries[i]->d_name);
    first = 0;
    free(entries[i]);
  }
  free(entries);
  printf("\n");
  fflush(stdout);
  return 0;
}

int builtin_rename(int argc, char **argv) {
  if (argc != 3) {
    return 1;
  }
  if (rename(argv[1], argv[2]) == -1) {
    int saved_errno = errno;
    perror("rename");
    return saved_errno;
  }
  return 0;
}

int builtin_unlink(int argc, char **argv) {
  if (argc != 2) {
    return 1;
  }
  if (unlink(argv[1]) == -1) {
    int saved_errno = errno;
    perror("unlink");
    return saved_errno;
  }
  return 0;
}

int builtin_remove(int argc, char **argv) {
  if (argc != 2) {
    return 1;
  }
  if (remove(argv[1]) == -1) {
    int saved_errno = errno;
    perror("remove");
    return saved_errno;
  }
  return 0;
}

int builtin_linkhard(int argc, char **argv) {
  if (argc != 3) {
    return 1;
  }
  if (link(argv[1], argv[2]) == -1) {
    int saved_errno = errno;
    perror("linkhard");
    return saved_errno;
  }
  return 0;
}

int builtin_linksoft(int argc, char **argv) {
  if (argc != 3) {
    return 1;
  }
  if (symlink(argv[1], argv[2]) == -1) {
    int saved_errno = errno;
    perror("linksoft");
    return saved_errno;
  }
  return 0;
}

int builtin_linkread(int argc, char **argv) {
  if (argc != 2) {
    return 1;
  }
  char buffer[PATH_MAX];
  ssize_t len = readlink(argv[1], buffer, sizeof(buffer) - 1);
  if (len == -1) {
    int saved_errno = errno;
    perror("linkread");
    return saved_errno;
  }
  buffer[len] = '\0';
  printf("%s\n", buffer);
  fflush(stdout);
  return 0;
}

int builtin_linklist(int argc, char **argv) {
  if (argc != 2) {
    return 1;
  }

  struct stat target_stat;
  if (stat(argv[1], &target_stat) == -1) {
    int saved_errno = errno;
    perror("linklist");
    return saved_errno;
  }

  DIR *dir = opendir(".");
  if (dir == NULL) {
    int saved_errno = errno;
    perror("linklist");
    return saved_errno;
  }

  struct dirent *entry;
  int first = 1;
  while ((entry = readdir(dir)) != NULL) {
    struct stat entry_stat;
    if (stat(entry->d_name, &entry_stat) == 0) {
      if (entry_stat.st_ino == target_stat.st_ino &&
          entry_stat.st_dev == target_stat.st_dev) {
        if (!first)
          printf("  ");
        printf("%s", entry->d_name);
        first = 0;
      }
    }
  }
  closedir(dir);
  printf("\n");
  fflush(stdout);
  return 0;
}

int builtin_cpcat(int argc, char **argv) {
  if (argc == 2) {
    FILE *src = fopen(argv[1], "rb");
    if (src == NULL) {
      int saved_errno = errno;
      perror("cpcat");
      return saved_errno;
    }

    char buffer[4096];
    size_t bytes;
    while ((bytes = fread(buffer, 1, sizeof(buffer), src)) > 0) {
      fwrite(buffer, 1, bytes, stdout);
    }

    if (ferror(src)) {
      int saved_errno = errno;
      perror("cpcat");
      fclose(src);
      return saved_errno;
    }

    fclose(src);
    fflush(stdout);
    return 0;
  } else if (argc == 3) {
    FILE *src = fopen(argv[1], "rb");
    if (src == NULL) {
      int saved_errno = errno;
      perror("cpcat");
      return saved_errno;
    }

    FILE *dst = fopen(argv[2], "wb");
    if (dst == NULL) {
      int saved_errno = errno;
      perror("cpcat");
      fclose(src);
      return saved_errno;
    }

    char buffer[4096];
    size_t bytes;
    while ((bytes = fread(buffer, 1, sizeof(buffer), src)) > 0) {
      if (fwrite(buffer, 1, bytes, dst) != bytes) {
        int saved_errno = errno;
        perror("cpcat");
        fclose(src);
        fclose(dst);
        return saved_errno;
      }
    }

    if (ferror(src)) {
      int saved_errno = errno;
      perror("cpcat");
      fclose(src);
      fclose(dst);
      return saved_errno;
    }

    fclose(src);
    fclose(dst);
    return 0;
  } else {
    return 1;
  }
}

int builtin_pid(int argc, char **argv) {
  printf("%d\n", getpid());
  fflush(stdout);
  return 0;
}

int builtin_ppid(int argc, char **argv) {
  printf("%d\n", getppid());
  fflush(stdout);
  return 0;
}

int builtin_uid(int argc, char **argv) {
  printf("%d\n", getuid());
  fflush(stdout);
  return 0;
}

int builtin_euid(int argc, char **argv) {
  printf("%d\n", geteuid());
  fflush(stdout);
  return 0;
}

int builtin_gid(int argc, char **argv) {
  printf("%d\n", getgid());
  fflush(stdout);
  return 0;
}

int builtin_egid(int argc, char **argv) {
  printf("%d\n", getegid());
  fflush(stdout);
  return 0;
}

int builtin_sysinfo(int argc, char **argv) {
  struct utsname info;
  if (uname(&info) == -1) {
    int saved_errno = errno;
    perror("sysinfo");
    return saved_errno;
  }

  printf("Sysname: %s\n", info.sysname);
  printf("Nodename: %s\n", info.nodename);
  printf("Release: %s\n", info.release);
  printf("Version: %s\n", info.version);
  printf("Machine: %s\n", info.machine);
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
      if (command && strcmp(command->name, "status") == 0) {
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
