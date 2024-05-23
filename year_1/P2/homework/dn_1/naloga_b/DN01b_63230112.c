#include <stdio.h>

void printInt(int num);
void printIntRec(int num);
int getLog();

int main() {
    int log = getLog();

    printInt(log);

    return 0;
}

int getLog() {
    int length = 0;
    int first_1_index = 0;
    char chr = getchar();

    while (chr != '\n') {
        length++;

        if (first_1_index == 0 && chr == '1') {
            first_1_index = length;
        }

        chr = getchar();
    }

    return length == first_1_index ?
        length - first_1_index :
        length + 1 - first_1_index;
}

void printInt(int num) {
    if (num == 0) {
        putchar('0');
        putchar('\n');
        return;
    }

    printIntRec(num);
    putchar('\n');
}

void printIntRec(int num) {
    if (num == 0) {
        return;
    }

    printIntRec(num / 10);

    putchar('0' + (num % 10));
}
