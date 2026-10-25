#include "vendor/jeaiii/jeaiii_to_text.h"
#include <cstddef>
#include <cstdint>
extern "C" size_t jeaiii_format(char *out, int64_t x) {
  return static_cast<size_t>(jeaiii::to_text_from_integer(out, x) - out);
}
