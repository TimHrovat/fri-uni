#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define BUFFER_SIZE 4096

int main(int argc, char *argv[]) {
  int input_fd, output_fd;
  ssize_t bytes_read, bytes_written;
  char buffer[BUFFER_SIZE];

  if (argc < 2 || strcmp(argv[1], "-") == 0) {
    input_fd = STDIN_FILENO;
  } else {
    input_fd = open(argv[1], O_RDONLY);
  }

  if (argc < 3) {
    output_fd = STDOUT_FILENO;
  } else {
    output_fd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644);
  }

  while ((bytes_read = read(input_fd, buffer, BUFFER_SIZE)) > 0) {
    char *ptr = buffer;
    ssize_t bytes_remaining = bytes_read;

    while (bytes_remaining > 0) {
      bytes_written = write(output_fd, ptr, bytes_remaining);
      ptr += bytes_written;
      bytes_remaining -= bytes_written;
    }
  }

  return 0;
}
