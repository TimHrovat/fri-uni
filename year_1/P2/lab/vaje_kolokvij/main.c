#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

struct Num {
    int value;
    struct Num *next;
} *start = NULL;

void insert(struct Num *num);
void print();
void delete();

int main() {
    int seq_len;

    scanf("%d", &seq_len);

    for (int i = 0; i < seq_len; i++) {
        struct Num *num = malloc(sizeof(struct Num *));

        scanf("%d", &num->value);

        num->next = NULL;

        insert(num);
    }

    scanf("%d", &seq_len);

    for (int i = 0; i < seq_len; i++) {
        struct Num *num = malloc(sizeof(struct Num *));

        scanf("%d", &num->value);

        num->next = NULL;

        insert(num);
    }

    print();
    delete();
}

void insert(struct Num *num) {
    struct Num *ptr_start = start;
    struct Num *prev = NULL;

    if (ptr_start == NULL) {
        start = num;

        return;
    }

    while (ptr_start != NULL) {
        if (ptr_start->value == num->value) {
            return;
        }

        if (ptr_start->value > num->value) {
            break;
        }

        prev = ptr_start;
        ptr_start = ptr_start->next;
    }

    if (prev == NULL) {
        num->next = start;
        start = num;
    } else {
        prev->next = num;
        num->next = ptr_start;
    }
}

void print() {
    struct Num *ptr_start = start;

    while (ptr_start != NULL) {
        printf("%d\n", ptr_start->value);

        ptr_start = ptr_start->next;
    }
}

void delete() {
    struct Num *ptr_start = start;
    struct Num *next;

    while (ptr_start != NULL) {
        next = ptr_start->next;

        free(ptr_start);

        ptr_start = next;
    }
}
