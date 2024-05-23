#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

bool is_valid_number(char *number, int len);
bool is_hex_digit(char c);
bool is_oct_digit(char c);
bool is_bin_digit(char c);
bool is_dec_digit(char c);

int main() {
    int len = 0;
    char chr;
    char *number = NULL;

    do {
        chr = getchar();
        number = realloc(number, (len + 1) * sizeof(char));

        if (chr == ' ' || chr == '\n') {
            number[len] = '\0';
            putchar(is_valid_number(number, len) ? '1' : '0');
            len = 0;
            continue;
        }

        number[len++] = chr;

    } while (chr != '\n');

    free(number);
    putchar('\n');

    return 0;
}

bool is_valid_number(char *number, int len) {
    // hex
    if (len > 2 && number[0] == '0' && number[1] == 'x') {
        for (int i = 2; i < len; i++) {
            if (!is_hex_digit(number[i])) return false;
        }
        return true;
    }

    // bin
    if (len > 2 && number[0] == '0' && number[1] == 'b') {
        for (int i = 2; i < len; i++) {
            if (!is_bin_digit(number[i])) return false;
        }
        return true;
    }

    // oct
    if (len > 1 && number[0] == '0') {
        for (int i = 1; i < len; i++) {
            if (!is_oct_digit(number[i])) return false;
        }
        return true;
    }

    // dec
    for (int i = 0; i < len; i++) {
        if (!is_dec_digit(number[i])) return false;
    }

    return true;
}

bool is_dec_digit(char c) { return (c >= '0' && c <= '9'); }

bool is_hex_digit(char c) { return (is_dec_digit(c) || (c >= 'A' && c <= 'F')); }

bool is_oct_digit(char c) { return (c >= '0' && c <= '7'); }

bool is_bin_digit(char c) { return c == '1' || c == '0'; }
