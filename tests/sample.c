#include <stdio.h>
#include <stdlib.h>

// A simple function with a potential buffer overflow vulnerability
void copy_buffer(char *input) {
    char buffer[10];
    // Vulnerable if input is larger than 10 bytes
    sprintf(buffer, "%s", input); 
    printf("Buffer contains: %s\n", buffer);
}

int main() {
    int x = 10;
    int y = 20;
    
    if (x < y) {
        printf("x is less than y\n");
    } else {
        printf("x is greater than or equal to y\n");
    }
    
    copy_buffer("This string is definitely too long for the buffer");
    
    return 0;
}
