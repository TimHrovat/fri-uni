#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// Napišite program, ki za vsak niz v vhodnem zaporedju izpiše 1, če ta niz
// predstavlja predznačeno desetiško konstanto, oziroma 0, če to ne drži.
// Predznačena desetiška konstanta zavzema eno od sledečih oblik:
// - Niz, ki ga sestavlja samo števka 0.
// - Niz, ki se prične z eno od števk z intervala [1, 9] in nadaljuje s poljubno
// dolgim zaporedjem števk z intervala [0, 9].
// - Niz, ki se prične z znakom + ali - in nadaljuje bodisi s števko 0 bodisi z
// zaporedjem, ki se prične z eno od števk z intervala [1, 9] in nadaljuje s
// poljubno dolgim zaporedjem števk z intervala [0, 9]
bool isValidChar(char chr, int length);
bool isNumeric(char chr);
bool isPlusMinus(char chr);
char isValidString(int length, char first_char, bool has_invalid_chars);

int main(int argc, char **argv) {
    char chr = getchar();
    int length = 0;
    char first_char;
    char second_char;
    bool has_invalid_chars = false;

    while (true) {
        if (chr != ' ' && has_invalid_chars) {
            chr = getchar();
            length++;
            continue;
        }

        if (chr == ' ' || chr == '\n') {
            putchar(isValidString(length, first_char, has_invalid_chars));
            chr = getchar();

            // reset
            length = 0;
            first_char = chr;
            has_invalid_chars = false;

            if (chr == '\n') {
                break;
            }

            continue;
        }

        if (length == 0) {
            first_char = chr;
        }

        if (!isValidChar(chr, length)) {
            has_invalid_chars = true;
        }

        length++;
        chr = getchar();
    }

    putchar('\n');

    return 0;
}

char isValidString(int length, char first_char, bool has_invalid_chars) {
    if (has_invalid_chars) {
        return '0';
    }

    if (length == 1 && isPlusMinus(first_char)) {
        return '0';
    }

    return '1';
}

bool isNumeric(char chr) {
    int numeric = chr - '0';

    return numeric >= 0 && numeric <= 9;
}

bool isValidChar(char chr, int length) { return isNumeric(chr) || isPlusMinus(chr) && length == 0; }

bool isPlusMinus(char chr) { return chr == '-' || chr == '+'; }
