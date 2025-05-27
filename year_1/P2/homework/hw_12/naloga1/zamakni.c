#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define INSET_WIDTH 4

char *strtrim(char *str);
void left_pad(char *str, int inset);

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("not enough arguments provided");
        return 0;
    }

    // input args
    char *input_file = argv[1];
    char *output_file = argv[2];

    // open file buffers
    FILE *fptr_in = fopen(input_file, "r");
    FILE *fptr_out = fopen(output_file, "w");

    // line buffer for input file
    char line[1000];
    int inset = 0;

    while (fgets(line, 1000, fptr_in)) {
        char *trimmed_line = strtrim(line);
        int len = strlen(trimmed_line);
        bool decrease_inset = trimmed_line[0] == '}' && inset >= 4;
        bool increase_inset = trimmed_line[len - 1] == '\n' && trimmed_line[len - 2] == '{';

        if (decrease_inset) {
            inset -= INSET_WIDTH;
        }

        left_pad(trimmed_line, inset);

        fprintf(fptr_out, "%s", trimmed_line);

        if (increase_inset) {
            inset += INSET_WIDTH;
        }
    }

    fclose(fptr_in);
    fclose(fptr_out);

    return 0;
}

char *strtrim(char *str) {
    char *_str = str;

    while (*_str == ' ') {
        _str++;
    }

    return _str;
}

void left_pad(char *str, int inset) {
    int len = strlen(str);

    for (int i = len; i >= 0; i--) {
        str[i + inset] = str[i];
    }

    for (int i = 0; i < inset; i++) {
        str[i] = ' ';
    }
}
