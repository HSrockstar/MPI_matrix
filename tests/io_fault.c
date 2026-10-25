#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
/* Test-only preload library. Affects only jobs explicitly launched with it. */
ssize_t pwrite(int fd, const void *buf, size_t bytes, off_t off) {
  static ssize_t (*real_write)(int, const void *, size_t, off_t);
  static int interrupted;
  if (!real_write) {
    void *symbol = dlsym(RTLD_NEXT, "pwrite");
    if (!symbol)
      _exit(99);
    _Static_assert(sizeof real_write == sizeof symbol, "POSIX dlsym ABI required");
    memcpy(&real_write, &symbol, sizeof real_write);
  }
  const char *mode = getenv("MATRIX_IO_FAULT");
  if (mode && !strcmp(mode, "zero"))
    return 0;
  if (mode && !strcmp(mode, "error")) {
    errno = ENOSPC;
    return -1;
  }
  if (mode && !strcmp(mode, "short-eintr")) {
    if (!interrupted++) {
      errno = EINTR;
      return -1;
    }
    if (bytes > 7)
      bytes = 7;
  }
  return real_write(fd, buf, bytes, off);
}
