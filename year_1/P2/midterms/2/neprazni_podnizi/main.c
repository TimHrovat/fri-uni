#include <stdio.h>
#include <string.h>

void generatePartitions(char *str, int n, int a, int b, int start, int parts, char *result, int resLen) {
    // Base case: if we've got at least `a` parts and `start` is at the end of the string
    if (parts >= a && start == n) {
        result[resLen - 1] = '\0'; // Replace last '|' with null character
        printf("%s\n", result);
        return;
    }
    
    // If the number of parts exceeds `b` or `start` is out of range, return
    if (parts > b || start >= n) {
        return;
    }

    // Try to partition the string by placing a '|' at different positions
    for (int i = start; i < n; ++i) {
        // Copy the current partition to result
        for (int j = start; j <= i; ++j) {
            result[resLen++] = str[j];
        }
        result[resLen++] = '|';
        
        // Recursive call to partition the rest of the string
        generatePartitions(str, n, a, b, i + 1, parts + 1, result, resLen);
        
        // Backtrack to try the next partitioning
        resLen -= (i - start + 2);
    }
}

int main() {
    char str[16];
    int a, b;

    // Read the input
    scanf("%s %d %d", str, &a, &b);
    
    int n = strlen(str);
    char result[32]; // Buffer to store the current partitioning result
    
    // Generate and print all partitions
    generatePartitions(str, n, a, b, 0, 1, result, 0);
    
    return 0;
}

