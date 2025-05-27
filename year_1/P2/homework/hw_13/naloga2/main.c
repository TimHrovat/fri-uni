#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int get_bit(unsigned char byte, int i) { return (byte >> (7 - i)) & 1; }

int main(int argc, char **argv) {
    FILE *file = fopen(argv[1], "rb");
    int p = atoi(argv[2]);
    int q = atoi(argv[3]);
    int p_byte = p / 8;
    int q_byte = ceil(q / 8.0);
    int num_of_bytes = q_byte - p_byte;
    unsigned char bytes[num_of_bytes];

    fseek(file, p_byte, sizeof(unsigned char));

    for (int i = 0; i < num_of_bytes; i++) {
        unsigned char byte[1];
        fread(byte, sizeof(unsigned char), 1, file);
        bytes[i] = byte[0];
    }

    int bits[num_of_bytes * 8];

    for (int i = 0; i < num_of_bytes; i++) {
        for (int j = 0; j < 8; j++) {
            bits[i * 8 + j] = get_bit(bytes[i], j);
        }
    }

    for (int i = p % 8; i < q - p_byte * 8; i++) {
        printf("%d", bits[i]);
    }

    fclose(file);
    return 0;
}
