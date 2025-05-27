#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct _Node {
    char line[10];
    struct _Node *prev;
} Node;

void reverse_print(Node *end, int n);

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("not enough args provided");
        return 0;
    }

    FILE *fptr = fopen(argv[1], "r");
    int output_c = strtol(argv[2], NULL, 10);

    Node *end = NULL;
    char line[10];

    while (fgets(line, 10, fptr)) {
        Node *node = malloc(sizeof(Node));
        node->prev = end;
        strcpy(node->line, line);

        end = node;
    }

    reverse_print(end, output_c);

    // clean up
    fclose(fptr);
    while (end != NULL) {
        Node *prev = end->prev;

        free(end);

        end = prev;
    }

    return 0;
}

void reverse_print(Node *end, int n) {
    if (n < 0) {
        return;
    }

    reverse_print(end->prev, n - 1);
    printf("%s", end->line);
}
