#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    FILE *file = fopen(argv[1], "rb");
    FILE *file_out = NULL;
    int k = atoi(argv[2]);

    unsigned char *buf = malloc(sizeof(unsigned char));

    long cnt = 0;
    while (fread(buf, sizeof(unsigned char), 1, file) == 1) {
        int index = cnt % k;
        int del = cnt / k;

        if (index == 0) {
            if (file_out != NULL) {
                fclose(file_out);
            }

            char fname[100];
            sprintf(fname, "datoteka.%d", del);

            file_out = fopen(fname, "wb");
        }

        fwrite(buf, sizeof(unsigned char), 1, file_out);

        cnt++;
    }

    if (file_out != NULL) {
        fclose(file_out);
    }

    free(buf);
    fclose(file);
    return 0;
}
